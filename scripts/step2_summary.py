"""
Step 2 comparison: evidence-span arms (global / scatter-8 / segment-4).

Reads checkpoints + history jsons, runs ONE val forward per checkpoint, and
reports per arm (mean over seeds):
  - AUROC (from history json, the registered performance readout)
  - concept-clarity metrics: C_feat (peak-feature consistency), frac below
    random baseline, H_pos (arm-aware position entropy), F_multi, C_seg
  - R (redundant_pair_frac, the registered representation attribute)

Interpretation ladder (registered in discussion):
  arm1 AND arm2 both improve  → candidate-count effect first
  only arm2 improves          → contiguous grouping adds value, but this does
                                NOT establish a short-timescale receptive
                                field (position RF stays 4.35s)

Usage:
    python scripts/step2_summary.py --split val
"""

import os, sys, argparse, json, warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.resnet1d import ResNet1D
from src.models.concept_bank import HMACDModel
from src.datasets.ptbxl_dataset import PTBXLDataset
from src.utils.concept_metrics import redundant_pair_frac, activation_validity_gate
from scripts.concept_mixing_check import (
    peak_feature_consistency,
    random_baseline_consistency,
    position_entropy,
    multi_peak_fraction,
    segment_correlation,
)

warnings.filterwarnings("ignore")

TOP_N = 50
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


@torch.no_grad()
def run_checkpoint(ckpt_path, history_path, loader, device):
    model, K = load_model(ckpt_path, device)
    all_sim, all_peak_pos, all_peak_feat, all_ecg = [], [], [], []
    for x, _ in loader:
        x = x.to(device)
        _, _, sim, features = model(x, return_features=True)
        peaks = sim.argmax(dim=-1)
        B = x.shape[0]
        b_idx = torch.arange(B).unsqueeze(1).expand(-1, K)
        peak_feat = features[b_idx, :, peaks].permute(0, 2, 1)
        all_sim.append(sim.cpu())
        all_peak_pos.append(peaks.cpu())
        all_peak_feat.append(peak_feat.cpu())
        all_ecg.append(x.cpu())
    sim = torch.cat(all_sim, dim=0).numpy()
    peak_pos = torch.cat(all_peak_pos, dim=0).numpy()
    peak_feat = torch.cat(all_peak_feat, dim=0).numpy()
    ecg = torch.cat(all_ecg, dim=0).numpy()
    N, K, L = sim.shape
    z = sim.max(axis=-1)  # [N, K]; masked entries are -inf so max is over allowed

    # allowed positions per concept (arm-aware)
    mask = getattr(model.concept_bank, "pos_mask", None)
    if mask is not None:
        mask_np = mask.cpu().numpy()
        allowed = [np.where(mask_np[k])[0] for k in range(K)]
    else:
        allowed = [np.arange(L) for _ in range(K)]

    baseline = random_baseline_consistency(peak_feat)

    per_concept = []
    for k in range(K):
        z_k = z[:, k]
        top_idx = np.argsort(z_k)[-TOP_N:][::-1]
        top_pos = peak_pos[top_idx, k]
        per_concept.append({
            "c_feat": peak_feature_consistency(peak_feat[:, k, :], top_idx),
            "h_pos": position_entropy(peak_pos[:, k], allowed[k]),
            "f_multi": multi_peak_fraction(sim[:, k, :]),
            "c_seg": segment_correlation(ecg, top_idx, top_pos),
        })

    c_feat = np.array([p["c_feat"] for p in per_concept])
    h_pos = np.array([p["h_pos"] for p in per_concept])
    f_multi = np.array([p["f_multi"] for p in per_concept])
    c_seg = np.array([p["c_seg"] for p in per_concept])

    # R (registered attribute) + validity gate on z
    R_dict = redundant_pair_frac(z, top_k=50, threshold_corr=0.80, threshold_jaccard=0.20)
    R = R_dict["frac"]
    gate = activation_validity_gate(z, near_const_std=1e-4, max_near_const_frac=0.05)

    # AUROC from history json (best val macro auroc)
    with open(history_path) as f:
        hist = json.load(f)
    entries = hist["history"] if isinstance(hist, dict) else hist
    best_auroc = max(e["val_auroc"] for e in entries)

    return {
        "c_feat_mean": float(c_feat.mean()),
        "c_feat_below_baseline_frac": float((c_feat < baseline).mean()),
        "baseline": float(baseline),
        "h_pos_mean": float(h_pos.mean()),
        "f_multi_mean": float(f_multi.mean()),
        "c_seg_mean": float(c_seg.mean()),
        "R": float(R),
        "gate_pass": bool(gate["passed"]),
        "near_const_frac": float(gate["near_const_frac"]),
        "val_auroc": best_auroc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = PTBXLDataset("/home/wuzuoxu/Data/ECG/1.0.3/", split=args.split, fold=10)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    arms = {
        "global (arm0)": [f"hmacd_k32_lc0.1_ld0.05_seed{s}_best.pt" for s in (42, 43, 44)],
        "scatter8 (arm1)": [f"hmacd_k32_lc0.1_ld0.05_sc8_seed{s}_best.pt" for s in (42, 43, 44)],
        "segment4 (arm2)": [f"hmacd_k32_lc0.1_ld0.05_sg4_seed{s}_best.pt" for s in (42, 43, 44)],
    }

    rows = []
    for arm, names in arms.items():
        for name in names:
            ckpt = os.path.join("results", name)
            hist = os.path.join("results", name.replace("_best.pt", "_history.json"))
            if not (os.path.exists(ckpt) and os.path.exists(hist)):
                print(f"SKIP {name} (missing ckpt or history)")
                continue
            print(f"analyzing {name} ...")
            r = run_checkpoint(ckpt, hist, loader, device)
            r["arm"] = arm
            r["run"] = name
            rows.append(r)

    df = pd.DataFrame(rows)
    out = "results/step2_summary.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*100}")
    print(f"Step 2 arm comparison — per-run values (split={args.split})")
    print(f"{'='*100}")
    cols = ["arm", "run", "val_auroc", "c_feat_mean", "c_feat_below_baseline_frac",
            "h_pos_mean", "f_multi_mean", "c_seg_mean", "R", "gate_pass"]
    print(df[cols].to_string(index=False))

    print(f"\n{'='*100}")
    print("Arm means (over seeds):")
    print(f"{'='*100}")
    agg = df.groupby("arm")[["val_auroc", "c_feat_mean", "c_feat_below_baseline_frac",
                              "h_pos_mean", "f_multi_mean", "c_seg_mean", "R"]].mean()
    print(agg.round(4).to_string())

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
