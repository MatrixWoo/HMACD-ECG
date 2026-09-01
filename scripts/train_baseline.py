"""
Train ResNet1D baseline on PTB-XL 5-superclass classification.

Usage:
    python scripts/train_baseline.py --config configs/baseline_resnet1d.yaml
"""

import argparse
import os
import sys
import yaml

import torch
from torch.utils.data import DataLoader

# ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.ptbxl_dataset import PTBXLDataset, SUPERCLASS_LIST
from src.models.resnet1d import ResNet1D
from src.training.trainer import train, evaluate
from src.utils.metrics import format_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_resnet1d.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # ---- load config ----
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["ptbxl_path"]
    batch_size = config["training"]["batch_size"]
    fold = config["data"].get("fold", 10)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data path: {data_path}")

    # ---- datasets ----
    train_ds = PTBXLDataset(data_path, split="train", fold=fold)
    val_ds = PTBXLDataset(data_path, split="val", fold=fold)
    test_ds = PTBXLDataset(data_path, split="test", fold=fold)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ---- model ----
    model = ResNet1D(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
    )

    # ---- train ----
    model, best_metrics = train(
        model, train_loader, val_loader, config,
        device=device, class_names=SUPERCLASS_LIST,
    )

    # ---- test ----
    print("\n========== Test Set Evaluation ==========")
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss, test_metrics, _, _ = evaluate(
        model, test_loader, criterion, device, class_names=SUPERCLASS_LIST
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(format_metrics(test_metrics, SUPERCLASS_LIST))


if __name__ == "__main__":
    main()
