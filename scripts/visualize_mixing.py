"""
Visual confirmation of concept mixing: stacked evidence-window plots.

For selected concepts, plot the raw 12-lead ECG evidence windows
(RF = 435 samples centered at each sample's peak position) for the
top-8 activating samples, aligned and same-scaled.

Usage:
    python scripts/visualize_mixing.py --concepts 29,6,7
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.resnet1d import ResNet1D
from src.models.concept_bank import HMACDModel
from src.datasets.ptbxl_dataset import PTBXLDataset, SUPERCLASS_LIST

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
RF = 435
STRIDE = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="results/hmacd_k32_lc0.1_ld0.05_seed42_best.pt")
    parser.add_argument("--data_path", type=str, default="/home/wuzuoxu/Data/ECG/1.0.3/")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--concepts", type=str, default="29,6,7")
    parser.add_argument("--top_n", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    concepts = [int(c) for c in args.concepts.split(",")]

    # load model
    ckpt = torch.load(args.ckpt, map_location=device)
    K = ckpt.get("config", {}).get("model", {}).get("num_concepts", 32)
    backbone = ResNet1D(in_channels=12, num_classes=5)
    model = HMACDModel(backbone, num_concepts=K, num_classes=5)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # data
    ds = PTBXLDataset(args.data_path, split=args.split, fold=10)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    # collect sim + ecg + labels
    all_sim, all_ecg, all_labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, _, sim = model(x)
            all_sim.append(sim.cpu().numpy())
            all_ecg.append(x.cpu().numpy())
            all_labels.append(y.numpy())
    sim = np.concatenate(all_sim, axis=0)
    ecg = np.concatenate(all_ecg, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    T = ecg.shape[-1]

    os.makedirs("figures/mixing", exist_ok=True)

    for k in concepts:
        z_k = sim[:, k].max(axis=-1)
        top_idx = np.argsort(z_k)[-args.top_n:][::-1]
        peak_pos = sim[top_idx, k].argmax(axis=-1)  # [top_n]

        n = args.top_n
        fig, axes = plt.subplots(n, 1, figsize=(14, 2.6 * n))
        fig.suptitle(f"Concept C{k:02d} — top-{n} evidence windows (RF=4.35s, peak-centered)",
                     fontsize=13, fontweight="bold")

        for row, (s, p) in enumerate(zip(top_idx, peak_pos)):
            ax = axes[row]
            c = int(p * STRIDE)
            lo = max(0, c - RF // 2)
            hi = min(T, lo + RF)
            lo = max(0, hi - RF)
            seg = ecg[s]  # [12, T]
            win = seg[:, lo:hi]  # [12, 435]
            span = win.max() - win.min()
            for lead in range(12):
                offset = (11 - lead) * 3.0
                ax.plot(win[lead] - win[lead].mean() + offset, linewidth=0.5, alpha=0.8)
            ax.axvline(c - lo, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
            true_lab = [SUPERCLASS_LIST[i] for i in range(5) if labels[s, i] > 0]
            ax.set_yticks([])
            ax.set_ylabel(f"#{s}\n{'/'.join(true_lab) if true_lab else 'NONE'}",
                          fontsize=7, rotation=0, ha="right", va="center")
            ax.set_xlim(0, RF)

        axes[-1].set_xlabel("time within evidence window (samples @100Hz; red line = peak center)")
        fig.tight_layout()
        fname = f"figures/mixing/concept_{k:02d}_windows.png"
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}  (peak positions: {peak_pos.tolist()})")


if __name__ == "__main__":
    main()
