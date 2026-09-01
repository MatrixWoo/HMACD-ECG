"""Training loop for ECG classification models."""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.metrics import compute_metrics, format_metrics


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Single training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, class_names=None):
    """Evaluate model. Returns (avg_loss, metrics_dict, y_true, y_probs)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_logits = []
    all_y = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_y.append(y.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_y, dim=0).numpy()
    # 安全 sigmoid: clip logits 避免 overflow
    logits = np.clip(logits, -100.0, 100.0)
    y_probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    metrics = compute_metrics(y_true, y_probs)
    return total_loss / n_batches, metrics, y_true, y_probs


def train(
    model,
    train_loader,
    val_loader,
    config,
    device=None,
    class_names=None,
):
    """
    Full training pipeline.

    Args:
        model: nn.Module
        train_loader: DataLoader
        val_loader: DataLoader
        config: dict with keys:
            training.lr, training.epochs, training.seed
        device: torch.device
        class_names: list of str (optional)

    Returns:
        model, best_metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = config["training"]
    lr = float(cfg.get("lr", 1e-3))
    epochs = int(cfg.get("epochs", 50))
    seed = int(cfg.get("seed", 42))

    set_seed(seed)

    patience = int(cfg.get("patience", 15))
    lr_factor = float(cfg.get("lr_factor", 0.5))
    lr_patience = int(cfg.get("lr_patience", 7))

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=lr_factor, patience=lr_patience, min_lr=1e-6
    )

    os.makedirs("results", exist_ok=True)
    exp_name = f"resnet1d_baseline_seed{seed}"
    best_ckpt_path = os.path.join("results", f"{exp_name}_best.pt")

    best_val_auroc = 0.0
    best_metrics = None
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics, _, _ = evaluate(model, val_loader, criterion, device, class_names)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(metrics["macro_auroc"])

        # scheduler step + early stopping
        val_auroc = metrics["macro_auroc"]
        scheduler.step(val_auroc)

        # best checkpoint by macro AUROC
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_metrics = metrics
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": val_auroc,
                    "config": config,
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"\n--- Epoch {epoch:3d}/{epochs} | Time: {elapsed:.0f}s ---")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(format_metrics(metrics, class_names))

    # ---- final: load best checkpoint ----
    elapsed = time.time() - t_start
    print(f"\n========== Training Finished ({elapsed:.0f}s) ==========")
    print(f"Best Val Macro AUROC: {best_val_auroc:.4f} @ epoch {best_epoch}, saved to {best_ckpt_path}")
    print(format_metrics(best_metrics, class_names))

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    return model, best_metrics
