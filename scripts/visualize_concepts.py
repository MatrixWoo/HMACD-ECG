"""
Visualize HMACD-ECG concept activations on ECG waveforms.

For each concept:
- Find the sample with highest z_k activation
- Locate the time window of peak similarity
- Plot 12-lead ECG with concept activation highlighted

Usage:
    python scripts/visualize_concepts.py \
        --ckpt results/hmacd_k32_seed42_best.pt \
        --concepts 0,3,14,29  # specific concepts to visualize
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.resnet1d import ResNet1D
from src.models.concept_bank import HMACDModel, compute_prototype_diversity
from src.datasets.ptbxl_dataset import PTBXLDataset, SUPERCLASS_LIST

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_model(ckpt_path, device):
    """Load trained HMACD model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    num_concepts = config.get("model", {}).get("num_concepts", 32)

    backbone = ResNet1D(in_channels=12, num_classes=5)
    model = HMACDModel(backbone, num_concepts=num_concepts, num_classes=5)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, num_concepts


def get_concept_activations(model, dataloader, device):
    """
    Run all samples through model, collect concept activations.
    Returns:
        all_z:      [N, K]  concept activations
        all_sim:    list of [K, L_i]  similarity maps (variable length → store as list)
        all_x:      list of [12, T_i] raw ECG signals
        all_labels: [N, 5]  labels
        all_logits: [N, 5]
    """
    all_z = []
    all_sim = []
    all_x = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits, z, sim = model(x)  # sim: [B, K, L]
            all_z.append(z.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(y)
            for i in range(x.shape[0]):
                all_sim.append(sim[i].cpu())      # [K, L]
                all_x.append(x[i].cpu())           # [12, T]

    all_z = torch.cat(all_z, dim=0).numpy()       # [N, K]
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_z, all_sim, all_x, all_labels, all_logits


def feature_pos_to_time(pos, feat_len, signal_len):
    """
    Map feature map position (0 .. feat_len-1) to signal time window.

    The backbone downsamples by factor of 32:
        stem (s=4) × layer2 (s=2) × layer3 (s=2) × layer4 (s=2) = 32x
    Each feature position covers ~32 time steps.
    """
    down_factor = signal_len / feat_len
    t_center = int(pos * down_factor + down_factor / 2)
    t_start = int(pos * down_factor)
    t_end = int(min((pos + 1) * down_factor, signal_len))
    return t_start, t_end, t_center


def plot_concept_card(concept_idx, top_samples, top_sims, top_positions,
                      all_x, all_z, all_labels, all_logits, save_dir,
                      class_names=SUPERCLASS_LIST):
    """
    Plot a concept card showing top activating samples with highlighted regions.

    Args:
        concept_idx:  int
        top_samples:  list of sample indices (sorted by z_k)
        top_sims:     list of similarity scores for each sample
        top_positions: list of peak positions for each sample
    """
    n_samples = len(top_samples)
    fig, axes = plt.subplots(n_samples, 2,
                              figsize=(18, 3.5 * n_samples),
                              gridspec_kw={"width_ratios": [3, 1]})
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, (s_idx, sim_val, pos) in enumerate(zip(top_samples, top_sims, top_positions)):
        x = all_x[s_idx]  # [12, T]
        z_k = all_z[s_idx, concept_idx]
        label_vec = all_labels[s_idx]
        logits_vec = all_logits[s_idx]
        probs = 1.0 / (1.0 + np.exp(-logits_vec))

        T = x.shape[-1]
        feat_len = sim_val.shape[0] if isinstance(sim_val, np.ndarray) else 1

        # ---- left panel: ECG with activation highlight ----
        ax_ecg = axes[row, 0]
        t_start, t_end, t_center = feature_pos_to_time(pos, feat_len, T)

        # plot all 12 leads
        lead_spacing = 3.0
        for lead in range(12):
            offset = (11 - lead) * lead_spacing
            signal = x[lead].numpy() + offset
            ax_ecg.plot(signal, color="royalblue", linewidth=0.5, alpha=0.7)

        # highlight activated region
        ax_ecg.axvspan(t_start, t_end, alpha=0.25, color="red", zorder=0)
        ax_ecg.axvline(t_center, color="darkred", linewidth=1, linestyle="--", alpha=0.7)

        # mark concept activation and class info
        active_classes = [class_names[c] for c in range(5) if label_vec[c] > 0]
        top_class = np.argmax(probs)
        ax_ecg.set_title(
            f"Concept {concept_idx:02d} | z={z_k:.3f} | "
            f"Labels: {','.join(active_classes) if active_classes else 'NONE'} | "
            f"Pred: {class_names[top_class]} ({probs[top_class]:.2f})",
            fontsize=11
        )
        ax_ecg.set_ylabel("Lead")
        ax_ecg.set_yticks([i * lead_spacing for i in range(12)])
        ax_ecg.set_yticklabels(LEAD_NAMES[::-1], fontsize=7)
        ax_ecg.set_xlabel(f"Time (samples) — highlight: [{t_start}, {t_end}]")

        # ---- right panel: similarity map + class probabilities ----
        ax_info = axes[row, 1]
        ax_info.axis("off")

        info_lines = []
        info_lines.append(f"Sample #{s_idx}")
        info_lines.append(f"Concept z_k = {z_k:.4f}")
        info_lines.append(f"Peak position = {pos} (t=[{t_start},{t_end}])")
        info_lines.append(f"Peak similarity = {sim_val.max():.4f}")
        info_lines.append("")
        info_lines.append("Class probabilities:")
        for c, name in enumerate(class_names):
            marker = "●" if label_vec[c] > 0 else "○"
            info_lines.append(f"  {marker} {name}: {probs[c]:.3f}")

        y_pos = 0.95
        for line in info_lines:
            ax_info.text(0.05, y_pos, line, transform=ax_info.transAxes,
                         fontsize=9, fontfamily="monospace", verticalalignment="top")
            y_pos -= 0.06

    fig.suptitle(f"Concept {concept_idx:02d} — Top {n_samples} Activating Samples",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    fname = os.path.join(save_dir, f"concept_{concept_idx:02d}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_concept_summary(all_z, all_labels, all_logits, save_dir,
                         class_names=SUPERCLASS_LIST):
    """Plot summary: z mean per class, concept-class correlation matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    probs = 1.0 / (1.0 + np.exp(-np.clip(all_logits, -100, 100)))

    # ---- left: z mean per class ----
    ax = axes[0]
    z_mean_per_class = np.zeros((5, all_z.shape[1]))
    for c in range(5):
        mask = all_labels[:, c] > 0
        if mask.sum() > 0:
            z_mean_per_class[c] = all_z[mask].mean(axis=0)

    im = ax.imshow(z_mean_per_class, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest")
    ax.set_xticks(range(0, all_z.shape[1], 4))
    ax.set_xticklabels([f"C{i:02d}" for i in range(0, all_z.shape[1], 4)], fontsize=7)
    ax.set_yticks(range(5))
    ax.set_yticklabels(class_names)
    ax.set_title("Mean Concept Activation per Class")
    ax.set_xlabel("Concept Index")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ---- right: concept-class Pearson r ----
    ax = axes[1]
    corr = np.zeros((all_z.shape[1], 5))
    for k in range(all_z.shape[1]):
        for c in range(5):
            if all_labels[:, c].std() > 0:
                corr[k, c] = np.corrcoef(all_z[:, k], all_labels[:, c])[0, 1]

    im = ax.imshow(corr.T, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                   vmin=-1, vmax=1)
    ax.set_xticks(range(0, all_z.shape[1], 4))
    ax.set_xticklabels([f"C{i:02d}" for i in range(0, all_z.shape[1], 4)], fontsize=7)
    ax.set_yticks(range(5))
    ax.set_yticklabels(class_names)
    ax.set_title("Concept-Label Pearson r")
    ax.set_xlabel("Concept Index")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("HMACD-ECG Concept Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fname = os.path.join(save_dir, "concept_summary.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="results/hmacd_k32_seed42_best.pt")
    parser.add_argument("--data_path", type=str,
                        default="/home/wuzuoxu/Data/ECG/1.0.3/")
    parser.add_argument("--concepts", type=str, default="all",
                        help="Comma-separated concept indices or 'all'")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top samples per concept")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max test samples to scan for top activations")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load model ----
    print(f"Loading model from {args.ckpt}...")
    model, K = load_model(args.ckpt, device)
    print(f"Model loaded: K={K} concepts")

    # ---- load test data ----
    ds = PTBXLDataset(args.data_path, split="test", fold=10)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test samples: {len(ds)}")

    # ---- extract activations ----
    print("Extracting concept activations...")
    all_z, all_sim, all_x, all_labels, all_logits = get_concept_activations(
        model, loader, device
    )
    N = all_z.shape[0]
    print(f"Extracted: z={all_z.shape}, x samples={len(all_x)}")

    # ---- diversity ----
    prototypes = model.concept_bank.prototypes.detach().cpu()
    div = compute_prototype_diversity(prototypes)
    print(f"Prototype diversity: {div:.4f}")

    # ---- save dir ----
    save_dir = "figures/concept_cards"
    os.makedirs(save_dir, exist_ok=True)

    # ---- concept summary plot ----
    fname = plot_concept_summary(all_z, all_labels, all_logits, save_dir)
    print(f"\nSaved: {fname}")

    # ---- per-concept cards ----
    if args.concepts == "all":
        concept_list = list(range(K))
    else:
        concept_list = [int(c) for c in args.concepts.split(",")]

    # limit scan to first max_samples for speed
    scan_N = min(N, args.max_samples)
    z_subset = all_z[:scan_N]

    print(f"\nGenerating concept cards for {len(concept_list)} concepts...")
    for k in concept_list:
        # find top activating samples for concept k
        z_k = z_subset[:, k]
        top_indices = np.argsort(z_k)[-args.top_k:][::-1]

        top_samples = []
        top_sims_list = []
        top_positions = []
        for idx in top_indices:
            sim_k = all_sim[idx][k]  # [L]
            top_pos = sim_k.argmax().item()
            top_samples.append(idx)
            top_sims_list.append(sim_k.numpy())
            top_positions.append(top_pos)

        fname = plot_concept_card(
            k, top_samples, top_sims_list, top_positions,
            all_x, all_z, all_labels, all_logits, save_dir
        )
        if k % 8 == 0 or k in concept_list[:5]:
            print(f"  Concept {k:02d}: z_max={z_k[top_indices[0]]:.3f} → {fname}")

    print(f"\nDone! All figures saved to {save_dir}/")
    print(f"  - concept_summary.png")
    print(f"  - concept_00.png ... concept_{K-1:02d}.png")


if __name__ == "__main__":
    main()
