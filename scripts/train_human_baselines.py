"""
Train human-feature-only baselines using PTB-XL+ features.

Models: Logistic Regression, Random Forest, XGBoost, MLP
Input:  PTB-XL+ features U [M=2025]
Output: 5-superclass multi-label classification

Answers: "How much diagnostic power do traditional ECG features alone provide?"
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.ptbxl_plus_dataset import PTBXLPlusDataset
from src.datasets.ptbxl_dataset import PTBXLDataset, SUPERCLASS_LIST
from src.utils.metrics import compute_metrics, format_metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

warnings.filterwarnings("ignore")


def load_data(data_path, plus_path, fold=10):
    """Load PTB-XL+ features and labels aligned with PTB-XL splits."""
    # feature datasets
    ds_train_f = PTBXLPlusDataset(data_path, plus_path, split="train", fold=fold)
    ds_val_f   = PTBXLPlusDataset(data_path, plus_path, split="val",   fold=fold)
    ds_test_f  = PTBXLPlusDataset(data_path, plus_path, split="test",  fold=fold)

    # label datasets (same splits)
    ds_train_l = PTBXLDataset(data_path, split="train", fold=fold)
    ds_val_l   = PTBXLDataset(data_path, split="val",   fold=fold)
    ds_test_l  = PTBXLDataset(data_path, split="test",  fold=fold)

    def build_matrix(feat_ds, label_ds):
        """Build aligned feature + label matrices."""
        X_list, y_list = [], []
        for i in range(len(feat_ds)):
            u = feat_ds[i].numpy()  # [M]
            _, y = label_ds[i]       # [5]
            X_list.append(u)
            y_list.append(y.numpy())
        return np.stack(X_list), np.stack(y_list)

    X_train, y_train = build_matrix(ds_train_f, ds_train_l)
    X_val,   y_val   = build_matrix(ds_val_f,   ds_val_l)
    X_test,  y_test  = build_matrix(ds_test_f,  ds_test_l)

    # standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Fit model on train, evaluate on val and test."""
    t0 = time.time()

    # fit
    model.fit(X_train, y_train)

    # predict
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)
        if isinstance(y_probs, list):
            # OneVsRest returns list of per-class proba
            y_probs = np.column_stack(y_probs)
        # ensure shape [N, 5]
        if y_probs.shape[1] != 5:
            # might be [N, 2] per class... handle edge cases
            y_probs = y_probs[:, 1] if y_probs.shape[1] == 2 else y_probs

        y_probs_val = model.predict_proba(X_val)
        if isinstance(y_probs_val, list):
            y_probs_val = np.column_stack(y_probs_val)
    else:
        # fallback: use decision_function
        y_probs = 1.0 / (1.0 + np.exp(-model.decision_function(X_test)))
        y_probs_val = 1.0 / (1.0 + np.exp(-model.decision_function(X_val)))

    # ensure binary label shape
    y_test = y_test.astype(np.int32)
    y_val = y_val.astype(np.int32)

    val_metrics = compute_metrics(y_val, y_probs_val)
    test_metrics = compute_metrics(y_test, y_probs)
    elapsed = time.time() - t0

    return {
        "name": name,
        "val_auroc": val_metrics["macro_auroc"],
        "test_auroc": test_metrics["macro_auroc"],
        "test_auprc": test_metrics["macro_auprc"],
        "test_f1": test_metrics["macro_f1"],
        "per_class_auroc": test_metrics["per_class_auroc"],
        "time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/home/wuzuoxu/Data/ECG/1.0.3/")
    parser.add_argument("--plus_path", type=str,
                        default="/home/wuzuoxu/Data/ECG/ptb-xl-plus/1.0.0/features/")
    parser.add_argument("--fold", type=int, default=10)
    parser.add_argument("--skip_xgboost", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Human-Feature-Only Baselines (Group 2)")
    print("=" * 60)
    print(f"Loading data from {args.data_path}...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
        args.data_path, args.plus_path, fold=args.fold
    )
    print(f"Features: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  X_val {X_val.shape}, X_test {X_test.shape}")
    print(f"  y_train sum per class: {y_train.sum(axis=0)}")

    results = []

    # ---- 1. Logistic Regression ----
    print("\n--- Logistic Regression ---")
    lr = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
        n_jobs=-1,
    )
    r = evaluate_model("LogisticRegression", lr,
                       X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(r)
    print(f"  Test AUROC: {r['test_auroc']:.4f}  ({r['time']:.0f}s)")

    # ---- 2. Random Forest ----
    print("\n--- Random Forest ---")
    rf = OneVsRestClassifier(
        RandomForestClassifier(n_estimators=200, max_depth=20,
                               min_samples_leaf=10, random_state=42,
                               n_jobs=-1),
        n_jobs=-1,
    )
    r = evaluate_model("RandomForest", rf,
                       X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(r)
    print(f"  Test AUROC: {r['test_auroc']:.4f}  ({r['time']:.0f}s)")

    # ---- 3. XGBoost ----
    if not args.skip_xgboost:
        print("\n--- XGBoost ---")
        try:
            from xgboost import XGBClassifier
            xgb = OneVsRestClassifier(
                XGBClassifier(n_estimators=200, max_depth=6,
                              learning_rate=0.1, subsample=0.8,
                              colsample_bytree=0.8, random_state=42,
                              verbosity=0, n_jobs=1),
                n_jobs=-1,
            )
            r = evaluate_model("XGBoost", xgb,
                               X_train, y_train, X_val, y_val, X_test, y_test)
            results.append(r)
            print(f"  Test AUROC: {r['test_auroc']:.4f}  ({r['time']:.0f}s)")
        except ImportError:
            print("  XGBoost not installed, skipping")

    # ---- 4. MLP ----
    print("\n--- MLP ---")
    mlp = OneVsRestClassifier(
        MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                      alpha=1e-4, batch_size=64, max_iter=200,
                      early_stopping=True, validation_fraction=0.1,
                      random_state=42, verbose=False),
    )
    r = evaluate_model("MLP", mlp,
                       X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(r)
    print(f"  Test AUROC: {r['test_auroc']:.4f}  ({r['time']:.0f}s)")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"{'Model':<22s} {'Test AUROC':>10s} {'Test AUPRC':>10s} {'Test F1':>10s} {'Time':>8s}")
    print(f"{'-'*70}")

    # also include ResNet1D baseline from saved results
    resnet_ckpt = "results/resnet1d_baseline_seed42_best.pt"
    if os.path.exists(resnet_ckpt):
        import torch
        ckpt = torch.load(resnet_ckpt, map_location="cpu")
        resnet_auroc = ckpt.get("val_auroc", float("nan"))
        print(f"{'ResNet1D (baseline)':<22s} {resnet_auroc:>10.4f} {'—':>10s} {'—':>10s} {'—':>8s}")

    for r in results:
        print(f"{r['name']:<22s} {r['test_auroc']:>10.4f} {r['test_auprc']:>10.4f} "
              f"{r['test_f1']:>10.4f} {r['time']:>7.0f}s")

        # per-class
        per_class = r.get("per_class_auroc", [])
        if per_class:
            cls_str = "  ".join(f"{n}:{v:.3f}" for n, v in zip(SUPERCLASS_LIST, per_class))
            print(f"  Per-class: {cls_str}")

    print(f"{'='*70}")

    # ---- save results ----
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/human_baselines.csv", index=False)
    print(f"\nResults saved to results/human_baselines.csv")


if __name__ == "__main__":
    main()
