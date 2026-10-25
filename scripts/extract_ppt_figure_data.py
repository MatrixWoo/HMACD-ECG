"""
Extract raw data for the 6 PPT figures from the frozen checkpoint (CPU-safe).

Outputs (results/ppt_figdata/):
  c14_top.json        — top-6 activating samples for C14: idx, z, peak pos, labels
  c14_ecg.npy         — [6, 12, 435] evidence windows (RF-window around peak)
  c14_intervention.json — MI-positive mean MI prob before/after C14 suppression
  c31_peaks.npy       — [N] C31 peak positions (val set)
  c28c31_overlap.json — top-50 sets for C28/C31, intersection, jaccard
  perf.json           — AUROC/F1 numbers for ResNet1D vs HMACD-ECG

Usage: python scripts/extract_ppt_figure_data.py [--device cpu]
"""

import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.resnet1d import ResNet1D
from src.models.concept_bank import HMACDModel
from src.datasets.ptbxl_dataset import PTBXLDataset, SUPERCLASS_LIST

STRIDE = 32
RF = 435


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    K = ckpt.get("config", {}).get("model", {}).get("num_concepts", 32)
    backbone = ResNet1D(in_channels=12, num_classes=5)
    model = HMACDModel(backbone, num_concepts=K, num_classes=5)
    sd = dict(ckpt["model_state_dict"])
    pos_mask = sd.pop("concept_bank.pos_mask", None)
    model.load_state_dict(sd)
    if pos_mask is not None:
        model.concept_bank.set_pos_mask(pos_mask)
    model.to(device).eval()
    return model, K


def evidence_window(ecg, peak_pos, T=1000):
    """435-sample window centered at the peak position (clamped)."""
    c = int(peak_pos * STRIDE)
    lo = max(0, c - RF // 2)
    hi = min(T, lo + RF)
    lo = max(0, hi - RF)
    return ecg[:, lo:hi]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_path", type=str, default="/home/wuzuoxu/Data/ECG/1.0.3/")
    parser.add_argument("--ckpt", type=str,
                        default="results/hmacd_k32_lc0.1_ld0.05_seed42_best.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    outdir = "results/ppt_figdata"
    os.makedirs(outdir, exist_ok=True)

    model, K = load_model(args.ckpt, device)
    ds = PTBXLDataset(args.data_path, split="val", fold=10)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    # ---- one forward pass: collect sim, z, labels, ecg ----
    all_sim, all_labels, all_ecg = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, _, sim = model(x)
            all_sim.append(sim.cpu())
            all_labels.append(y)
            all_ecg.append(x.cpu())
    sim = torch.cat(all_sim, dim=0).numpy()      # [N, K, L]
    labels = torch.cat(all_labels, dim=0).numpy()
    ecg = torch.cat(all_ecg, dim=0).numpy()
    N = sim.shape[0]
    z = sim.max(axis=-1)
    print(f"val: N={N}, sim={sim.shape}")

    # ---- C14: top-6 activating samples ----
    k14 = 14
    top6 = np.argsort(z[:, k14])[-6:][::-1]
    c14_top = []
    c14_ecg_list = []
    for s in top6:
        p = int(sim[s, k14].argmax())
        c14_top.append({
            "idx": int(s),
            "z": float(z[s, k14]),
            "peak_pos": p,
            "labels": [SUPERCLASS_LIST[c] for c in range(5) if labels[s, c] > 0],
        })
        c14_ecg_list.append(evidence_window(ecg[s], p))
    json.dump(c14_top, open(f"{outdir}/c14_top.json", "w"), indent=2)
    np.save(f"{outdir}/c14_ecg.npy", np.array(c14_ecg_list))
    print(f"C14 top6: {[(t['idx'], round(t['z'],3)) for t in c14_top]}")

    # ---- C14 intervention: MI-positive samples, mean MI prob before/after ----
    mi_mask = labels[:, 1] > 0
    mi_idx = np.where(mi_mask)[0]
    # baseline probs
    all_probs_base, all_probs_supp = [], []
    with torch.no_grad():
        for start in range(0, len(mi_idx), 128):
            batch_idx = mi_idx[start:start + 128]
            xb = torch.from_numpy(ecg[batch_idx]).to(device)
            lb, _, _ = model(xb)
            all_probs_base.append(torch.sigmoid(lb)[:, 1].cpu())
            ls, _, _ = model(xb, intervene_concepts=[14])
            all_probs_supp.append(torch.sigmoid(ls)[:, 1].cpu())
    prob_base = torch.cat(all_probs_base).numpy()
    prob_supp = torch.cat(all_probs_supp).numpy()
    inter = {
        "n_mi_positive": int(len(mi_idx)),
        "mean_mi_prob_original": float(prob_base.mean()),
        "mean_mi_prob_suppress_c14": float(prob_supp.mean()),
        "mean_drop": float((prob_base - prob_supp).mean()),
        "std_drop": float((prob_base - prob_supp).std()),
    }
    json.dump(inter, open(f"{outdir}/c14_intervention.json", "w"), indent=2)
    print(f"C14 intervention: {inter}")

    # ---- C31 peak positions (full val set) ----
    c31_peaks = sim[:, 31].argmax(axis=-1)
    np.save(f"{outdir}/c31_peaks.npy", c31_peaks)
    near_end = ((c31_peaks >= 29) | (c31_peaks <= 2)).mean()
    print(f"C31 peaks: near-end frac = {near_end:.3f}")

    # ---- C16 (canonical boundary-sensitive concept on the current ckpt) ----
    # NOTE: the 2026-07 checkpoint's C31 boundary story does not hold after the
    # 2026-09 retrain; C16 (83% near-end) is the stable representative. The
    # boundary-sensitive INDEX SET {C12,C13,C16,C17,C01,C28} appears in both
    # trainings.
    c16_peaks = sim[:, 16].argmax(axis=-1)
    np.save(f"{outdir}/c16_peaks.npy", c16_peaks)
    c16_near = ((c16_peaks >= 29) | (c16_peaks <= 2)).mean()
    c16_corr = [float(np.corrcoef(z[:, 16], labels[:, c])[0, 1]) for c in range(5)]
    # intervention on all val samples
    pb_all, ps_all = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=128, shuffle=False, num_workers=0):
            x = x.to(device)
            lb, _, _ = model(x)
            ls, _, _ = model(x, intervene_concepts=[16])
            pb_all.append(torch.sigmoid(lb).cpu())
            ps_all.append(torch.sigmoid(ls).cpu())
    pb = torch.cat(pb_all).numpy()
    ps = torch.cat(ps_all).numpy()
    c16_meta = {
        "near_end_frac": float(c16_near),
        "peak_hist": np.bincount(c16_peaks, minlength=sim.shape[-1]).tolist(),
        "label_corr": c16_corr,
        "intervention_delta": [float((pb[:, c] - ps[:, c]).mean()) for c in range(5)],
    }
    json.dump(c16_meta, open(f"{outdir}/c16_meta.json", "w"), indent=2)
    print(f"C16: near-end {c16_near:.3f}, corr {c16_corr}, "
          f"interv Δ {c16_meta['intervention_delta']}")

    # ---- C28/C31 top-50 overlap ----
    top50 = {k: set(np.argsort(z[:, k])[-50:][::-1].tolist())
             for k in (28, 31)}
    inter_set = top50[28] & top50[31]
    union = top50[28] | top50[31]
    ov = {"n": 50, "inter_size": len(inter_set), "union_size": len(union),
          "jaccard": len(inter_set) / len(union)}
    json.dump(ov, open(f"{outdir}/c28c31_overlap.json", "w"), indent=2)
    print(f"C28/C31 overlap: {ov}")

    # ---- performance numbers (from recorded results, not recomputed) ----
    perf = {
        "resnet1d": {"auroc": 0.9183, "f1": 0.7100},
        "hmacd": {"auroc": 0.9134, "f1": 0.7168},
        "source": "test split, seed42, recorded 2026-07-15 experiment logs",
    }
    json.dump(perf, open(f"{outdir}/perf.json", "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()
