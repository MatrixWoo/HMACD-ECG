"""
Train HMACD-ECG model on PTB-XL 5-superclass classification.
Supports multi-GPU via DataParallel.

Usage:
    # single GPU
    python scripts/train_hmacd.py --config configs/hmacd_resnet1d.yaml

    # multi-GPU
    python scripts/train_hmacd.py --config configs/hmacd_resnet1d.yaml --gpu_ids 0,1,2,3
"""

import argparse
import os
import sys
import yaml
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.ptbxl_dataset import PTBXLDataset, SUPERCLASS_LIST
from src.models.resnet1d import ResNet1D
from src.models.concept_bank import (
    HMACDModel,
    concept_losses,
    compute_prototype_diversity,
)
from src.training.trainer import set_seed
from src.utils.metrics import format_metrics, compute_metrics


def get_model_prototypes(model):
    """Get prototypes, handling DataParallel wrapper."""
    return model.module.get_prototypes() if isinstance(model, nn.DataParallel) else model.get_prototypes()


def train_epoch(model, dataloader, optimizer, config, device):
    """Single training epoch with concept losses."""
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_compact = 0.0
    total_diverse = 0.0
    n_batches = 0

    lambda_compact = float(config["training"].get("lambda_compact", 0.1))
    lambda_diverse = float(config["training"].get("lambda_diverse", 0.05))

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, z, sim = model(x)
        prototypes = get_model_prototypes(model)

        loss, loss_dict = concept_losses(
            logits, y, sim, prototypes,
            lambda_compact=lambda_compact,
            lambda_diverse=lambda_diverse,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss_dict["total"]
        total_cls += loss_dict["cls_loss"]
        total_compact += loss_dict["compactness"]
        total_diverse += loss_dict["diversity"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls / n_batches,
        "compactness": total_compact / n_batches,
        "diversity": total_diverse / n_batches,
    }


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    """Evaluate HMACD model. Returns (avg_loss, metrics)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits, all_y = [], []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits, z, sim = model(x)
        cls_loss = criterion(logits, y)

        total_loss += cls_loss.item()
        n_batches += 1
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())

    logits_np = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_y, dim=0).numpy()
    logits_np = np.clip(logits_np, -100.0, 100.0)
    y_probs = 1.0 / (1.0 + np.exp(-logits_np))
    metrics = compute_metrics(y_true, y_probs)
    return total_loss / n_batches, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hmacd_resnet1d.yaml")
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated GPU IDs, e.g. '0,1,2,3'")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # ---- parse GPUs ----
    gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
    n_gpus = len(gpu_ids)
    main_device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    device = main_device  # alias for function args
    print(f"Using GPUs: {gpu_ids}  (n_gpus={n_gpus})")

    # ---- load config ----
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["ptbxl_path"]
    batch_size_per_gpu = int(config["training"]["batch_size"])
    fold = config["data"].get("fold", 10)
    print(f"Batch size per GPU: {batch_size_per_gpu}  "
          f"(effective: {batch_size_per_gpu * n_gpus})")

    # ---- datasets ----
    train_ds = PTBXLDataset(data_path, split="train", fold=fold)
    val_ds = PTBXLDataset(data_path, split="val", fold=fold)
    test_ds = PTBXLDataset(data_path, split="test", fold=fold)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size_per_gpu, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size_per_gpu * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size_per_gpu * 2, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ---- model ----
    cfg = config["training"]
    lr = float(cfg.get("lr", 0.001))
    epochs = int(cfg.get("epochs", 50))
    seed = int(cfg.get("seed", 42))
    patience = int(cfg.get("patience", 15))
    lr_patience = int(cfg.get("lr_patience", 7))
    lr_factor = float(cfg.get("lr_factor", 0.5))
    num_concepts = int(config["model"].get("num_concepts", 32))

    set_seed(seed)

    backbone = ResNet1D(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
    )
    model = HMACDModel(backbone, num_concepts=num_concepts,
                       num_classes=config["model"]["num_classes"])
    model = model.to(main_device)

    # ---- multi-GPU ----
    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"DataParallel wrapped over {n_gpus} GPUs: {gpu_ids}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.2f}M  |  K={num_concepts}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_factor, patience=lr_patience, min_lr=1e-6,
    )

    os.makedirs("results", exist_ok=True)
    exp_name = f"hmacd_k{num_concepts}_seed{seed}"
    best_ckpt_path = os.path.join("results", f"{exp_name}_best.pt")

    best_val_auroc = 0.0
    best_metrics = None
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training HMACD-ECG: K={num_concepts} λ_compact={config['training'].get('lambda_compact',0.1)} λ_diverse={config['training'].get('lambda_diverse',0.05)}")
    print(f"{'='*60}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # ---- train ----
        train_info = train_epoch(model, train_loader, optimizer, config, device)

        # ---- validate ----
        val_loss, metrics = eval_epoch(model, val_loader, criterion, device)
        val_auroc = metrics["macro_auroc"]

        # ---- schedule ----
        scheduler.step(val_auroc)

        # ---- checkpoint ----
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_metrics = metrics
            best_epoch = epoch
            patience_counter = 0
            # save unwrapped model state
            sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": sd,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": val_auroc,
                    "config": config,
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1

        # ---- report ----
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            div = compute_prototype_diversity(get_model_prototypes(model))
            print(f"\n--- Epoch {epoch:3d}/{epochs} | Time: {elapsed:.0f}s | LR: {optimizer.param_groups[0]['lr']:.2e} ---")
            print(f"  Train: loss={train_info['loss']:.4f} cls={train_info['cls_loss']:.4f} "
                  f"compact={train_info['compactness']:.4f} diverse={div:.4f}")
            print(f"  Val:   loss={val_loss:.4f}  AUROC={val_auroc:.4f}")
            print(format_metrics(metrics, SUPERCLASS_LIST))

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    elapsed = time.time() - t_start

    # =====================================================================
    #  STAGE 1: Training Complete
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 1: Training Complete ({elapsed:.0f}s, {best_epoch} epochs)")
    print(f"{'='*60}")
    print(f"Best Val Macro AUROC: {best_val_auroc:.4f} @ epoch {best_epoch}")
    print(format_metrics(best_metrics, SUPERCLASS_LIST))

    # ---- load best checkpoint ----
    ckpt = torch.load(best_ckpt_path, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    # =====================================================================
    #  STAGE 2: Concept Quality Analysis
    # =====================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: Concept Quality Analysis")
    print(f"{'='*60}")

    prototypes = get_model_prototypes(model)  # [K, C]
    diversity = compute_prototype_diversity(prototypes)
    print(f"Prototype Diversity: {diversity:.4f} (mean off-diag cosine sim)")

    # concept activation statistics on test set
    model.eval()
    all_z = []
    all_logits = []
    all_y = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits, z, sim = model(x)
            all_z.append(z.cpu())
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())

    z_all = torch.cat(all_z, dim=0).numpy()  # [N, K]
    logits_all = torch.cat(all_logits, dim=0).numpy()
    y_all = torch.cat(all_y, dim=0).numpy()

    # per-concept activation statistics
    z_mean = z_all.mean(axis=0)      # [K]
    z_std = z_all.std(axis=0)        # [K]
    z_activation_rate = (z_all > 0).mean(axis=0)  # [K]

    print(f"\nConcept activation summary:")
    print(f"  z mean range: [{z_mean.min():.3f}, {z_mean.max():.3f}]")
    print(f"  z std  range: [{z_std.min():.3f}, {z_std.max():.3f}]")
    print(f"  Activation rate: [{z_activation_rate.min():.3f}, {z_activation_rate.max():.3f}]")

    # concept-class correlation
    probs_all = 1.0 / (1.0 + np.exp(-np.clip(logits_all, -100, 100)))
    z_class_corr = np.zeros((z_all.shape[1], y_all.shape[1]))  # [K, 5]
    for k in range(z_all.shape[1]):
        for c in range(y_all.shape[1]):
            if y_all[:, c].std() > 0:
                z_class_corr[k, c] = np.corrcoef(z_all[:, k], y_all[:, c])[0, 1]

    print(f"\nTop-3 concepts per class (by Pearson r):")
    for c, cls_name in enumerate(SUPERCLASS_LIST):
        top_k = np.argsort(np.abs(z_class_corr[:, c]))[-3:][::-1]
        corr_str = "  ".join(f"C{k:02d}:{z_class_corr[k, c]:+.3f}" for k in top_k)
        print(f"  {cls_name:>6s}: {corr_str}")

    # =====================================================================
    #  STAGE 3: Test Set Evaluation
    # =====================================================================
    print(f"\n{'='*60}")
    print("STAGE 3: Test Set Evaluation")
    print(f"{'='*60}")
    test_loss, test_metrics = eval_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(format_metrics(test_metrics, SUPERCLASS_LIST))

    # =====================================================================
    #  Summary
    # =====================================================================
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Config:         {args.config}")
    print(f"  K concepts:     {num_concepts}")
    print(f"  GPUs:           {gpu_ids}")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best Val AUROC: {best_val_auroc:.4f}")
    print(f"  Test AUROC:     {test_metrics['macro_auroc']:.4f}")
    print(f"  Proto Diversity:{diversity:.4f}")
    print(f"  Checkpoint:     {best_ckpt_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
