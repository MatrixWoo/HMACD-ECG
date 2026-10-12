"""
Train HMACD-ECG model on PTB-XL 5-superclass classification.
Supports multi-GPU via DataParallel.

Usage:
    # single GPU
    python scripts/train_hmacd.py --config configs/hmacd_resnet1d.yaml

    # multi-GPU
    python scripts/train_hmacd.py --config configs/hmacd_resnet1d.yaml --gpu_ids 0,1,2,3

    # the single final test evaluation (development runs default to --eval_split val)
    python scripts/train_hmacd.py --config configs/hmacd_resnet1d.yaml --eval_split test

STAGE 2 (concept statistics) and STAGE 3 (evaluation) both run on `--eval_split`,
which defaults to "val". With the default, `test_ds` is never constructed, so a
development run cannot read the test set even by accident. Use 'test' only for
the unified evaluation after the selection protocol is frozen (proposal §7.3).
Note: this test set already carries exploratory use (proposal §6.3), so the
final evaluation is reported as exploratory, not as a first-ever look.
"""

import argparse
import json
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
    build_pos_mask,
)
from src.training.trainer import set_seed
from src.utils.metrics import format_metrics, compute_metrics
from src.utils.exp_name import experiment_name


def get_model_prototypes(model):
    """Get prototypes, handling DataParallel wrapper."""
    return model.module.get_prototypes() if isinstance(model, nn.DataParallel) else model.get_prototypes()


def wandb_safe_config(config):
    """Config copy safe to upload: strips the local data path.

    wandb uploads whatever `config=` receives to the user's account. The base
    yaml contains `ptbxl_path` (a private server path) -- it must not leave the
    machine. Only the path is stripped; all training/model fields stay.
    """
    cfg = json.loads(json.dumps(config))  # deep copy via json round-trip
    cfg.get("data", {}).pop("ptbxl_path", None)
    return cfg


def regularization_family(config):
    """Which diversity family a config belongs to (matches the Gate A arms)."""
    train = config.get("training", {})
    if float(train.get("lambda_decor", 0.0)) > 0:
        return "decorrelation"
    if float(train.get("lambda_diverse", 0.0)) > 0:
        return "mean_cosine"
    return "no_diversity"


def train_epoch(model, dataloader, optimizer, config, device):
    """Single training epoch with concept losses."""
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_compact = 0.0
    total_diverse = 0.0
    total_decor = 0.0
    n_batches = 0

    lambda_compact = float(config["training"].get("lambda_compact", 0.1))
    lambda_diverse = float(config["training"].get("lambda_diverse", 0.05))
    lambda_decor = float(config["training"].get("lambda_decor", 0.0))

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, z, sim = model(x)
        prototypes = get_model_prototypes(model)

        loss, loss_dict = concept_losses(
            logits, y, z, sim, prototypes,
            lambda_compact=lambda_compact,
            lambda_diverse=lambda_diverse,
            lambda_decor=lambda_decor,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss_dict["total"]
        total_cls += loss_dict["cls_loss"]
        total_compact += loss_dict["compactness"]
        total_diverse += loss_dict["diversity"]
        total_decor += loss_dict.get("decorrelation", 0.0)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls / n_batches,
        "compactness": total_compact / n_batches,
        "diversity": total_diverse / n_batches,
        "decorrelation": total_decor / n_batches,
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
    parser.add_argument("--eval_split", choices=["val", "test"], default="val",
                        help="Split used by STAGE 2 (concept statistics) and STAGE 3 "
                             "(evaluation). Default 'val' keeps development runs off "
                             "the test set entirely -- test_ds is not even built. "
                             "Use 'test' only for the single final evaluation.")
    parser.add_argument("--wandb", dest="use_wandb", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Log scalars to wandb (default on; falls back silently "
                             "if wandb is not installed or not logged in). "
                             "--no-wandb disables.")
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
    eval_split = args.eval_split

    train_ds = PTBXLDataset(data_path, split="train", fold=fold)
    val_ds = PTBXLDataset(data_path, split="val", fold=fold)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size_per_gpu, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size_per_gpu * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # The test set is built ONLY when explicitly asked for. Building it lazily --
    # rather than building it and remembering not to use it -- makes accidental
    # reads impossible instead of merely avoided.
    eval_loader = val_loader
    eval_n = len(val_ds)
    if eval_split == "test":
        test_ds = PTBXLDataset(data_path, split="test", fold=fold)
        eval_loader = DataLoader(test_ds, batch_size=batch_size_per_gpu * 2,
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)
        eval_n = len(test_ds)
        print("!" * 70)
        print("!! --eval_split test: this run READS THE TEST SET.")
        print("!! Only for the single final evaluation after the selection protocol")
        print("!! is frozen. Development runs must use the default (val).")
        print("!" * 70)
    print(f"STAGE 2/3 will evaluate on: {eval_split} ({eval_n} samples)")

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

    # ---- evidence-span arm (Step 2 of the temporal-granularity line) ----
    # arm="global" keeps the historical behavior; "scatter"/"segment" restrict
    # which positions each concept may pool over (same candidate count).
    # Set BEFORE the DataParallel wrap so the buffer replicates to all GPUs.
    arm = config["model"].get("arm", "global")
    if arm != "global":
        seg_len = int(config["data"].get("segment_length", 1000))
        with torch.no_grad():
            dummy = torch.zeros(1, config["model"]["in_channels"], seg_len,
                                device=main_device)
            _, _, _, dummy_feat = model(dummy, return_features=True)
            L = dummy_feat.shape[-1]
        mask = build_pos_mask(
            arm, num_concepts, L,
            arm_positions=int(config["model"].get("arm_positions", 8)),
            segment_m=int(config["model"].get("segment_m", 4)),
        ).to(main_device)
        model.concept_bank.set_pos_mask(mask)
        n_allowed = int(mask.sum(dim=1)[0].item())
        print(f"Evidence-span arm: {arm} — {n_allowed}/{L} positions per concept")

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
    # name must cover every swept dimension, or runs at fixed K/seed overwrite
    # each other when the loss weights vary (see src/utils/exp_name.py)
    exp_name = experiment_name(config)
    best_ckpt_path = os.path.join("results", f"{exp_name}_best.pt")
    history_path = os.path.join("results", f"{exp_name}_history.json")

    # ---- wandb (optional, scalars only, local paths stripped) ----
    use_wandb = args.use_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="hmacd-ecg", name=exp_name,
                       group=regularization_family(config),
                       config=wandb_safe_config(config))
        except Exception as e:
            # not installed, not logged in, or no network: train without it.
            # The history json is always written, so nothing is lost.
            print(f"[wandb] disabled: {type(e).__name__}: {e}")
            use_wandb = False

    best_val_auroc = 0.0
    best_metrics = None
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"\n{'='*60}")
    print(f"Training HMACD-ECG: K={num_concepts} λ_compact={config['training'].get('lambda_compact',0.1)} λ_diverse={config['training'].get('lambda_diverse',0.05)} λ_decor={config['training'].get('lambda_decor',0.0)}")
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

        # ---- persist per-epoch metrics (proposal Phase 0 item 9) ----
        # written every epoch so a killed run still leaves its trajectory; the
        # file is tiny (a few KB for 50 epochs).
        history.append({
            "epoch": epoch,
            "train": train_info,
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "val": metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.time() - t_start,
            "is_best": bool(val_auroc >= best_val_auroc and patience_counter == 0),
        })
        with open(history_path, "w") as f:
            json.dump({"config": config, "history": history}, f, indent=2)

        # ---- wandb: scalars only (no waveforms, no patient ids) ----
        if use_wandb:
            try:
                wandb.log({
                    "train/loss": train_info["loss"],
                    "train/cls_loss": train_info["cls_loss"],
                    "train/compactness": train_info["compactness"],
                    "train/diversity": train_info["diversity"],
                    "train/decorrelation": train_info["decorrelation"],
                    "val/loss": val_loss,
                    "val/macro_auroc": val_auroc,
                    "val/micro_auroc": metrics["micro_auroc"],
                    **{f"val/auroc_{cls}": a for cls, a in
                       zip(SUPERCLASS_LIST, metrics["per_class_auroc"])},
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_s": time.time() - t_start,
                })
            except Exception as e:
                # a mid-run network blip must not kill training
                print(f"[wandb] log failed, disabling: {type(e).__name__}: {e}")
                use_wandb = False

        # ---- report ----
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            # end-of-epoch snapshot; note history's 'diversity' entry is the
            # per-batch training average, so the two can legitimately differ
            div = compute_prototype_diversity(get_model_prototypes(model))
            print(f"\n--- Epoch {epoch:3d}/{epochs} | Time: {elapsed:.0f}s | LR: {optimizer.param_groups[0]['lr']:.2e} ---")
            decor_str = ""
            if float(config["training"].get("lambda_decor", 0.0)) > 0:
                decor_str = f" decor={train_info['decorrelation']:.4f}"
            print(f"  Train: loss={train_info['loss']:.4f} cls={train_info['cls_loss']:.4f} "
                  f"compact={train_info['compactness']:.4f} diverse={div:.4f}{decor_str}")
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
    print(f"STAGE 2: Concept Quality Analysis (split={eval_split})")
    print(f"{'='*60}")

    prototypes = get_model_prototypes(model)  # [K, C]
    diversity = compute_prototype_diversity(prototypes)
    print(f"Prototype Diversity: {diversity:.4f} (mean off-diag cosine sim)")

    # concept activation statistics on the evaluation split (val by default)
    model.eval()
    all_z = []
    all_logits = []
    all_y = []
    with torch.no_grad():
        for x, y in eval_loader:
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
    #  STAGE 3: Evaluation on the selected split
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 3: {eval_split.capitalize()} Set Evaluation")
    print(f"{'='*60}")
    eval_loss, eval_metrics = eval_epoch(model, eval_loader, criterion, device)
    print(f"{eval_split.capitalize()} Loss: {eval_loss:.4f}")
    print(format_metrics(eval_metrics, SUPERCLASS_LIST))
    if eval_split == "val":
        # Same checkpoint, same deterministic loader (shuffle=False, model.eval()),
        # so this must reproduce STAGE 1. Tolerance 1e-4 absorbs GPU conv
        # nondeterminism; a real bug (wrong loader or checkpoint) shows up far
        # larger than that.
        delta = abs(eval_metrics["macro_auroc"] - best_val_auroc)
        flag = "OK" if delta < 1e-4 else "MISMATCH -- check loader/checkpoint"
        print(f"  [self-check] {flag}: re-evaluating val on the best checkpoint "
              f"gives {eval_metrics['macro_auroc']:.6f} vs best "
              f"{best_val_auroc:.6f} (delta {delta:.2e})")

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
    print(f"  Eval split:     {eval_split} ({eval_n} samples)")
    print(f"  {eval_split.capitalize()} AUROC:     {eval_metrics['macro_auroc']:.4f}")
    print(f"  Proto Diversity:{diversity:.4f}")
    print(f"  Checkpoint:     {best_ckpt_path}")
    print(f"{'='*60}")

    if use_wandb:
        wandb.log({
            "final/best_val_auroc": best_val_auroc,
            "final/best_epoch": best_epoch,
            f"final/{eval_split}_auroc": eval_metrics["macro_auroc"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
