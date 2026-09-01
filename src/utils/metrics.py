"""Multi-label classification metrics for ECG diagnosis."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)


def compute_metrics(y_true, y_probs, threshold=0.5):
    """
    Compute multi-label classification metrics.

    Args:
        y_true:  np.ndarray [N, C]  ground-truth binary labels
        y_probs: np.ndarray [N, C]  predicted probabilities (after sigmoid)
        threshold: float             binarization threshold for F1/Accuracy

    Returns:
        dict with keys:
            macro_auroc, macro_auprc, micro_auroc, weighted_auroc,
            per_class_auroc (list),
            macro_f1, micro_f1, accuracy,
            per_class_f1 (list)
    """
    N, C = y_true.shape
    metrics = {}

    # ---- AUROC ----
    try:
        metrics["macro_auroc"] = float(
            roc_auc_score(y_true, y_probs, average="macro")
        )
    except ValueError:
        metrics["macro_auroc"] = float("nan")

    try:
        metrics["micro_auroc"] = float(
            roc_auc_score(y_true, y_probs, average="micro")
        )
    except ValueError:
        metrics["micro_auroc"] = float("nan")

    try:
        metrics["weighted_auroc"] = float(
            roc_auc_score(y_true, y_probs, average="weighted")
        )
    except ValueError:
        metrics["weighted_auroc"] = float("nan")

    # ---- per-class AUROC ----
    per_class = []
    for c in range(C):
        if y_true[:, c].sum() == 0 or y_true[:, c].sum() == N:
            per_class.append(float("nan"))
        else:
            per_class.append(float(roc_auc_score(y_true[:, c], y_probs[:, c])))
    metrics["per_class_auroc"] = per_class

    # ---- AUPRC ----
    try:
        metrics["macro_auprc"] = float(
            average_precision_score(y_true, y_probs, average="macro")
        )
    except ValueError:
        metrics["macro_auprc"] = float("nan")

    # ---- F1 (with threshold) ----
    y_pred = (y_probs >= threshold).astype(np.int32)
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    per_class_f1 = []
    for c in range(C):
        per_class_f1.append(
            float(f1_score(y_true[:, c], y_pred[:, c], zero_division=0))
        )
    metrics["per_class_f1"] = per_class_f1

    # ---- Accuracy (exact match & subset) ----
    metrics["exact_match"] = float(accuracy_score(y_true, y_pred))
    metrics["subset_accuracy"] = float(
        (y_true == y_pred).all(axis=1).sum() / N
    )

    return metrics


def format_metrics(metrics, class_names=None):
    """Pretty-print metrics dict."""
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(metrics["per_class_auroc"]))]

    lines = []
    lines.append(f"  Macro AUROC:  {metrics['macro_auroc']:.4f}")
    lines.append(f"  Micro AUROC:  {metrics['micro_auroc']:.4f}")
    lines.append(f"  Macro AUPRC:  {metrics['macro_auprc']:.4f}")
    lines.append(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    lines.append(f"  Micro F1:     {metrics['micro_f1']:.4f}")
    lines.append(f"  Exact Match:  {metrics['exact_match']:.4f}")
    lines.append(f"  Subset Acc:   {metrics['subset_accuracy']:.4f}")
    lines.append("  Per-class AUROC:")
    for name, auroc in zip(class_names, metrics["per_class_auroc"]):
        lines.append(f"    {name:>6s}: {auroc:.4f}")
    lines.append("  Per-class F1:")
    for name, f1 in zip(class_names, metrics["per_class_f1"]):
        lines.append(f"    {name:>6s}: {f1:.4f}")

    return "\n".join(lines)
