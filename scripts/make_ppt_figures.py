"""
Generate the 6 HMACD-ECG figures for the HKUST interview PPT.

All numbers come from recorded experiment results or from the frozen
checkpoint via scripts/extract_ppt_figure_data.py (results/ppt_figdata/).
Style: academic, clean, white background, validated categorical palette
(blue #2a78d6 / orange #eb6834 / aqua #1baf7a — dataviz skill slots 1-3).

NOTE on spurious concept: the 2026-07 checkpoint's C31 boundary story
(61.5% near-end, Jaccard 0.887 with C28) does NOT hold on the current
(2026-09 retrained) checkpoint. The boundary-sensitive set {C16, C13,
C17, C12, C01} is stable across both trainings; the figure uses C16
(83.4% near-end) with current numbers.

Usage: python scripts/make_ppt_figures.py
"""

import os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---- palette (dataviz skill reference, light mode) ----
C_BLUE = "#2a78d6"     # slot 1 — ResNet1D
C_ORANGE = "#eb6834"   # slot 2 — HMACD-ECG
C_AQUA = "#1baf7a"     # slot 3
C_CRIT = "#d03b3b"     # status critical
C_GOOD = "#0ca30c"     # status good
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"

FIGDIR = "figures/ppt"
DATADIR = "results/ppt_figdata"
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_json(name):
    with open(os.path.join(DATADIR, name)) as f:
        return json.load(f)


# ============================================================
# 1. performance_bar_auroc_f1.png
# ============================================================
def fig_performance():
    perf = load_json("perf.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    methods = ["ResNet1D", "HMACD-ECG"]
    for ax, metric, key, ylo, yhi in [
        (axes[0], "Macro AUROC", "auroc", 0.86, 0.94),
        (axes[1], "Macro F1", "f1", 0.60, 0.80),
    ]:
        vals = [perf["resnet1d"][key], perf["hmacd"][key]]
        colors = [C_BLUE, C_ORANGE]
        bars = ax.bar(methods, vals, color=colors, width=0.55,
                      edgecolor="none", zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (yhi - ylo) * 0.012,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=13,
                    fontweight="bold", color=INK)
        # delta annotation
        d = vals[0] - vals[1]
        ax.text(0.5, ylo + (yhi - ylo) * 0.08,
                f"Δ = {d:+.4f}" + (" (no sacrifice)" if key == "auroc" else ""),
                ha="center", fontsize=10, color=INK2)
        ax.set_ylim(ylo, yhi)
        ax.set_title(metric, fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(axis="x", labelsize=12)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Concept bottleneck preserves classification performance",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/performance_bar_auroc_f1.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved performance_bar_auroc_f1.png")


# ============================================================
# 2. concept_card_C14.png
# ============================================================
def fig_concept_card():
    top = load_json("c14_top.json")
    ecg = np.load(f"{DATADIR}/c14_ecg.npy")  # [6, 12, 435]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    lead_off = 2.6
    for i, ax in enumerate(axes.flat):
        seg = ecg[i]
        t = np.arange(seg.shape[-1]) / 100.0  # seconds
        for lead in range(12):
            offset = (11 - lead) * lead_off
            ax.plot(t, seg[lead] - seg[lead].mean() + offset,
                    linewidth=0.55, color=C_BLUE, alpha=0.85)
        ax.axvline(seg.shape[-1] / 2 / 100.0, color=C_CRIT, linewidth=1.0,
                   linestyle="--", alpha=0.8)
        labels = "/".join(top[i]["labels"]) if top[i]["labels"] else "—"
        ax.set_title(f"#{top[i]['idx']}  z={top[i]['z']:.3f}  [{labels}]",
                     fontsize=9.5, color=INK)
        ax.set_yticks([])
        ax.set_ylim(-lead_off * 2, 12 * lead_off)
        ax.set_xlim(0, seg.shape[-1] / 100.0)
        if i >= 3:
            ax.set_xlabel("time (s)", fontsize=8, color=MUTED)
    fig.suptitle("Concept C14 — top-activating ECG segments (evidence windows, 12-lead)",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.005,
             "All top-6 activating samples carry an MI label · z ≈ 0.91 · "
             "corr(z_C14, MI) = +0.62 · suppressing C14 drops MI probability by 9.2pp",
             ha="center", fontsize=10, color=INK2)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(f"{FIGDIR}/concept_card_C14.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("saved concept_card_C14.png")


# ============================================================
# 3. concept_intervention_C14.png
# ============================================================
def fig_intervention():
    inter = load_json("c14_intervention.json")
    before = inter["mean_mi_prob_original"]
    after = inter["mean_mi_prob_suppress_c14"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars = ax.bar(["Original", "Suppress C14"], [before, after],
                  color=[C_BLUE, C_ORANGE], width=0.5, zorder=3)
    for b, v in zip(bars, [before, after]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=14, fontweight="bold")
    # drop annotation
    ax.annotate("", xy=(1, after), xytext=(1, before),
                arrowprops=dict(arrowstyle="-|>", color=C_CRIT, lw=1.6))
    ax.text(1.16, (before + after) / 2, f"−{100 * (before - after):.1f}pp",
            fontsize=13, fontweight="bold", color=C_CRIT, va="center")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Mean P(MI) on MI-positive samples", fontsize=11)
    ax.set_title("Suppressing one concept changes the diagnosis",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.grid(axis="x", visible=False)
    fig.text(0.5, 0.01, f"N = {inter['n_mi_positive']} MI-positive val samples · paired mean drop "
              f"{inter['mean_drop']:.3f} ± {inter['std_drop']:.3f}",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(f"{FIGDIR}/concept_intervention_C14.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved concept_intervention_C14.png")


# ============================================================
# 4. has_scatter_rf.png
# ============================================================
def fig_has():
    import pandas as pd
    has_df = pd.read_excel("results/has_results.xlsx",
                           sheet_name="1-HAS-Summary", skiprows=2)
    has_rf = has_df["HAS(RF)"].values[:32].astype(float)
    order = np.argsort(has_rf)
    med = np.median(has_rf)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    colors = [C_ORANGE if v < med else C_BLUE for v in has_rf[order]]
    ax.bar([f"C{i:02d}" for i in order], has_rf[order], color=colors,
           width=0.7, zorder=3)
    ax.axhline(med, color=C_CRIT, linewidth=1.4, linestyle="--", zorder=4)
    ax.text(31.2, med + 0.015, f"median = {med:.2f}", color=C_CRIT,
            fontsize=11, fontweight="bold", ha="right")
    ax.set_ylim(0.35, 1.0)
    ax.set_ylabel("HAS(RF)  =  R²(z_k, g(U))", fontsize=11)
    ax.set_xlabel("concept")
    ax.set_title("Every concept is substantially predictable from traditional ECG features",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.grid(axis="x", visible=False)
    fig.text(0.5, 0.005,
             "U = 2,025 PTB-XL+ features (PR/QRS/QT/ST-T…) · Random Forest regressor · "
             "median 0.76, range [0.51, 0.88]",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(f"{FIGDIR}/has_scatter_rf.png", dpi=220, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("saved has_scatter_rf.png")


# ============================================================
# 5. spurious_concept_C16.png
# ============================================================
def fig_spurious():
    meta = load_json("c16_meta.json")
    hist = np.array(meta["peak_hist"], dtype=float)
    hist /= hist.sum()
    corr = np.array(meta["label_corr"])
    class_names = ["NORM", "MI", "STTC", "CD", "HYP"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- left: peak position distribution, boundary regions shaded ----
    ax = axes[0]
    pos = np.arange(len(hist))
    ax.bar(pos, hist, color=C_BLUE, width=0.8, zorder=3)
    ax.axvspan(-0.5, 2.5, color=C_CRIT, alpha=0.12, zorder=2)
    ax.axvspan(28.5, 31.5, color=C_CRIT, alpha=0.12, zorder=2)
    ax.text(15.5, hist.max() * 0.95, "83.4% of peaks at\nrecord boundaries",
            ha="center", fontsize=11.5, fontweight="bold", color=C_CRIT)
    ax.set_xlabel("peak position (32 temporal bins, 10 s record)")
    ax.set_ylabel("fraction of val samples")
    ax.set_title("C16 fires at the record edges", fontsize=12.5,
                 fontweight="bold", pad=8)
    ax.grid(axis="x", visible=False)

    # ---- right: weak label association — not a reliable disease concept ----
    ax = axes[1]
    bars = ax.bar(class_names, corr, color=[C_ORANGE if v < 0 else C_BLUE for v in corr],
                  width=0.55, zorder=3)
    ax.axhline(0, color=INK2, linewidth=1.0)
    for b, v in zip(bars, corr):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.015 if v >= 0 else -0.03),
                f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, color=INK)
    ax.axhline(0.62, color=C_GOOD, linewidth=1.2, linestyle="--")
    ax.text(4.35, 0.62, "C14–MI corr\n(+0.62, for reference)",
            fontsize=8.5, color=C_GOOD, ha="right", va="bottom")
    ax.set_ylim(-0.5, 0.75)
    ax.set_ylabel("corr(z_C16, label)")
    ax.set_title("…but its label association is weak", fontsize=12.5,
                 fontweight="bold", pad=8)
    ax.grid(axis="x", visible=False)

    fig.suptitle("Some concepts are not reliable: boundary sensitivity of C16",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, 0.005,
             "val split, frozen checkpoint (seed 42) · the boundary-sensitive set "
             "{C12, C13, C16, C17, C01, C28} appears in both the 2026-07 and "
             "2026-09 trainings",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(f"{FIGDIR}/spurious_concept_C16.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved spurious_concept_C16.png")


# ============================================================
# 6. hmacd_pipeline_clean.png
# ============================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    boxes = [
        (0.2, "12-lead ECG\n[B, 12, 1000]", C_BLUE),
        (2.7, "ResNet1D\nencoder", C_BLUE),
        (5.2, "Prototype\nconcept bank\nK = 32", C_ORANGE),
        (7.7, "Concept activations\nz  [B, 32]", C_ORANGE),
        (10.2, "Classification head\nŷ  [B, 5]", C_AQUA),
    ]
    w, h = 2.15, 1.55
    y = 1.1
    for x, label, color in boxes:
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.08,rounding_size=0.12",
                             linewidth=1.6, edgecolor=color,
                             facecolor="white", zorder=3)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10.5, color=INK, zorder=4)
    # arrows
    for x1, x2 in [(2.35, 2.7), (4.85, 5.2), (7.35, 7.7), (9.85, 10.2)]:
        ax.add_patch(FancyArrowPatch((x1, y + h / 2), (x2, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.5, color=INK2, zorder=2))
    # concept analysis annotation under bank
    ax.text(6.275, 0.42, "z_k = max_l cos(h_l, p_k)   ·   evidence span & "
             "receptive field govern what a concept sees",
            ha="center", fontsize=9.5, color=INK2)
    fig.suptitle("HMACD-ECG: concept-bottleneck ECG diagnosis",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/hmacd_pipeline_clean.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved hmacd_pipeline_clean.png")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    fig_performance()
    fig_concept_card()
    fig_intervention()
    fig_has()
    fig_spurious()
    fig_pipeline()
    print("done: 6 figures in figures/ppt/")
