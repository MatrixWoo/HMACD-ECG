"""
Step 1 of concept-mixing check: do the top-activating segments of ONE
prototype repeatedly express SIMILAR patterns, or mixed ones?

For each concept k (frozen checkpoint, val split):
  1. Peak-feature consistency — pairwise cosine sim of backbone features
     at the top-N peak positions across samples (with random baseline)
  2. Peak-position entropy / spread — are peaks localized or scattered?
  3. Multi-peak fraction — does one sample fire the same concept at
     several DISTINCT regions (evidence of within-record mixing)?
  4. Label homogeneity — do top activators share a dominant class?
  5. Segment correlation — pairwise Pearson r of raw ECG evidence windows

Usage:
    python scripts/concept_mixing_check.py \
        --ckpt results/hmacd_k32_lc0.1_ld0.05_seed42_best.pt \
        --split val --top_n 50
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.resnet1d import ResNet1D
from src.models.concept_bank import HMACDModel
from src.datasets.ptbxl_dataset import PTBXLDataset, SUPERCLASS_LIST

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
RF_SAMPLES = 435       # theoretical encoder receptive field (see receptive_field.py)
STRIDE = 32            # total downsampling of the encoder

warnings.filterwarnings("ignore")


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    K = ckpt.get("config", {}).get("model", {}).get("num_concepts", 32)
    backbone = ResNet1D(in_channels=12, num_classes=5)
    model = HMACDModel(backbone, num_concepts=K, num_classes=5)
    # arm checkpoints store a pos_mask buffer; a fresh ConceptBank has none,
    # so a strict load would reject the key. Pop it, load the rest, then
    # re-register it via set_pos_mask (same semantics, verified by the assert).
    sd = dict(ckpt["model_state_dict"])
    mask_key = "concept_bank.pos_mask"
    pos_mask = sd.pop(mask_key, None)
    model.load_state_dict(sd)
    if pos_mask is not None:
        model.concept_bank.set_pos_mask(pos_mask)
    model.to(device).eval()
    return model, K


@torch.no_grad()
def collect_activations(model, dataloader, device):
    """
    Collect per-sample: sim [K, L], peak positions [K], peak features [K, C],
    labels [5], raw ECG [12, T]. Returns lists keyed by concept.
    """
    all_sim, all_peak_pos, all_peak_feat, all_labels = [], [], [], []
    all_ecg = []

    for x, y in dataloader:
        x = x.to(device)
        _, _, sim, features = model(x, return_features=True)  # sim [B,K,L], feat [B,C,L]
        peaks = sim.argmax(dim=-1)                             # [B, K]
        B, C, L = features.shape
        # gather feature vector at each concept's peak position
        # features[b, :, peaks[b,k]] → [B, K, C]
        b_idx = torch.arange(B).unsqueeze(1).expand(-1, peaks.shape[1])
        peak_feat = features[b_idx, :, peaks].permute(0, 2, 1)  # [B, K, C]
        all_sim.append(sim.cpu())
        all_peak_pos.append(peaks.cpu())
        all_peak_feat.append(peak_feat.cpu())
        all_labels.append(y)
        all_ecg.append(x.cpu())

    sim = torch.cat(all_sim, dim=0).numpy()          # [N, K, L]
    peak_pos = torch.cat(all_peak_pos, dim=0).numpy()  # [N, K]
    peak_feat = torch.cat(all_peak_feat, dim=0).numpy()  # [N, K, C]
    labels = torch.cat(all_labels, dim=0).numpy()      # [N, 5]
    ecg = torch.cat(all_ecg, dim=0).numpy()            # [N, 12, T]
    return sim, peak_pos, peak_feat, labels, ecg


def peak_feature_consistency(peak_feat, top_idx, n_pairs=200):
    """Mean pairwise cosine similarity of peak features among top samples."""
    feats = peak_feat[top_idx]  # [topN, C]
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    # sample a bounded number of pairs
    m = len(feats)
    if m < 2:
        return float("nan")
    rng = np.random.default_rng(42)
    pairs = rng.integers(0, m, size=(min(n_pairs, m * (m - 1) // 2), 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    cos = (feats[pairs[:, 0]] * feats[pairs[:, 1]]).sum(axis=1)
    return float(cos.mean())


def random_baseline_consistency(peak_feat, n_samples=2000, n_pairs=200):
    """Same metric but on randomly paired (sample, concept) features."""
    N, K, C = peak_feat.shape
    rng = np.random.default_rng(7)
    idx = rng.integers(0, N, size=(n_pairs, 2))
    feats = peak_feat.reshape(-1, C)
    f1 = feats[idx[:, 0] * K]  # arbitrary concepts per sample — fine for baseline
    f2 = feats[idx[:, 1] * K]
    f1 = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8)
    f2 = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8)
    return float((f1 * f2).sum(axis=1).mean())


def position_entropy(positions, allowed_positions):
    """Normalized entropy of peak positions over the concept's ALLOWED set.

    positions: absolute peak positions (the mask already constrains these).
    allowed_positions: sorted absolute positions the concept may use.
    0 = always the same allowed position, 1 = uniform over the allowed set.
    (Global arm: allowed_positions = all L positions — unchanged semantics.)
    """
    n_allowed = len(allowed_positions)
    if n_allowed <= 1:
        return 0.0
    pos_map = {p: i for i, p in enumerate(allowed_positions)}
    remapped = np.array([pos_map[int(p)] for p in positions])
    counts = np.bincount(remapped, minlength=n_allowed).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(n_allowed))


def multi_peak_fraction(sim_k, min_gap=4, rel_thresh=0.85):
    """Fraction of samples with >=2 well-separated local maxima within rel_thresh of max."""
    n_multi = 0
    for s in sim_k:
        peak = s.max()
        above = np.where(s >= rel_thresh * peak)[0]
        # cluster consecutive positions
        clusters = []
        for p in above:
            if clusters and p - clusters[-1][-1] <= min_gap:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        if len(clusters) >= 2:
            n_multi += 1
    return n_multi / len(sim_k)


def segment_correlation(ecg, top_samples, top_positions, n_pairs=200):
    """Pairwise Pearson r of raw 12-lead evidence windows among top samples."""
    segs = []
    for s, p in zip(top_samples, top_positions):
        T = ecg.shape[-1]
        c = int(p * STRIDE)  # window center in input samples
        lo = max(0, c - RF_SAMPLES // 2)
        hi = min(T, lo + RF_SAMPLES)
        lo = max(0, hi - RF_SAMPLES)
        seg = ecg[s, :, lo:hi].reshape(-1)  # 12-lead flattened
        segs.append(seg)
    segs = np.array(segs)
    m = len(segs)
    if m < 2:
        return float("nan")
    rng = np.random.default_rng(42)
    pairs = rng.integers(0, m, size=(min(n_pairs, m * (m - 1) // 2), 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    rs = [np.corrcoef(segs[i], segs[j])[0, 1] for i, j in pairs]
    return float(np.mean(rs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="results/hmacd_k32_lc0.1_ld0.05_seed42_best.pt")
    parser.add_argument("--data_path", type=str, default="/home/wuzuoxu/Data/ECG/1.0.3/")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load model + data ----
    model, K = load_model(args.ckpt, device)
    print(f"Model: {args.ckpt}  (K={K})")
    ds = PTBXLDataset(args.data_path, split=args.split, fold=10)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    print(f"Split: {args.split}, N={len(ds)}")

    # ---- collect ----
    print("Collecting activations...")
    sim, peak_pos, peak_feat, labels, ecg = collect_activations(model, loader, device)
    N, K, L = sim.shape
    print(f"sim={sim.shape}, peak_pos={peak_pos.shape}, ecg={ecg.shape}")

    baseline = random_baseline_consistency(peak_feat)
    print(f"\nRandom-pairing baseline for peak-feature consistency: {baseline:.4f}")

    # ---- allowed positions per concept (arm-aware) ----
    mask = getattr(model.concept_bank, "pos_mask", None)
    if mask is not None:
        mask_np = mask.cpu().numpy()
        allowed_by_concept = [np.where(mask_np[k])[0] for k in range(K)]
        print(f"pos_mask present: {[len(a) for a in allowed_by_concept[:1]]} positions "
              f"per concept (arm checkpoint)")
    else:
        allowed_by_concept = [np.arange(L) for _ in range(K)]
        print("no pos_mask: global arm (all L positions allowed)")

    # ---- per-concept metrics ----
    rows = []
    for k in range(K):
        z_k = sim[:, k].max(axis=-1)          # [N]
        top_idx = np.argsort(z_k)[-args.top_n:][::-1]
        top_pos = peak_pos[top_idx, k]

        c_feat = peak_feature_consistency(peak_feat[:, k, :], top_idx)
        h_pos = position_entropy(peak_pos[:, k], allowed_by_concept[k])
        pos_std_ms = top_pos.std() * STRIDE * 10  # samples → ms
        f_multi = multi_peak_fraction(sim[:, k, :])
        # label homogeneity: dominant class presence among top samples
        top_labels = labels[top_idx]
        dom_class = int(top_labels.mean(axis=0).argmax())
        label_frac = float(top_labels[:, dom_class].mean())
        c_seg = segment_correlation(ecg, top_idx, top_pos)

        rows.append({
            "concept": k,
            "z_top_mean": float(z_k[top_idx].mean()),
            "c_feat": c_feat,
            "c_feat_over_baseline": c_feat - baseline,
            "H_pos": h_pos,
            "pos_std_ms": pos_std_ms,
            "F_multi": f_multi,
            "label_frac": label_frac,
            "dominant_class": SUPERCLASS_LIST[dom_class],
            "c_seg": c_seg,
        })

    df = pd.DataFrame(rows)

    # ---- report ----
    print(f"\n{'='*95}")
    print(f"Concept mixing report (top_n={args.top_n}, RF={RF_SAMPLES}samples={RF_SAMPLES/100:.2f}s)")
    print(f"{'='*95}")
    print(f"{'C':>3s} {'z̄_top':>7s} {'C_feat':>7s} {'Δbl':>6s} {'H_pos':>6s} "
          f"{'pos_std':>8s} {'F_multi':>8s} {'labFrac':>8s} {'dom':>5s} {'C_seg':>6s}")
    print("-" * 95)
    for _, r in df.iterrows():
        print(f"C{r['concept']:02d} {r['z_top_mean']:>7.3f} {r['c_feat']:>7.4f} "
              f"{r['c_feat_over_baseline']:>+6.3f} {r['H_pos']:>6.3f} "
              f"{r['pos_std_ms']:>8.0f} {r['F_multi']:>8.2f} "
              f"{r['label_frac']:>8.2f} {r['dominant_class']:>5s} {r['c_seg']:>6.3f}")

    print(f"\nSummary statistics:")
    print(f"  C_feat:   mean={df['c_feat'].mean():.4f}  min={df['c_feat'].min():.4f}  max={df['c_feat'].max():.4f}  (random baseline {baseline:.4f})")
    print(f"  H_pos:    mean={df['H_pos'].mean():.4f}  (0=fully localized, 1=uniform)")
    print(f"  F_multi:  mean={df['F_multi'].mean():.4f}  max={df['F_multi'].max():.4f}")
    print(f"  C_seg:    mean={df['c_seg'].mean():.4f}  min={df['c_seg'].min():.4f}")

    # concepts sorted by consistency (ascending = most mixed)
    worst = df.sort_values("c_feat").head(8)
    print(f"\nMost-mixed concepts (lowest peak-feature consistency):")
    for _, r in worst.iterrows():
        print(f"  C{r['concept']:02d}: C_feat={r['c_feat']:.3f}, H_pos={r['H_pos']:.3f}, "
              f"F_multi={r['F_multi']:.2f}, dom={r['dominant_class']}")

    best = df.sort_values("c_feat", ascending=False).head(8)
    print(f"\nMost-consistent concepts:")
    for _, r in best.iterrows():
        print(f"  C{r['concept']:02d}: C_feat={r['c_feat']:.3f}, H_pos={r['H_pos']:.3f}, "
              f"F_multi={r['F_multi']:.2f}, dom={r['dominant_class']}")

    # ---- save ----
    out = f"results/concept_mixing_{os.path.basename(args.ckpt).replace('_best.pt','')}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
