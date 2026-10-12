"""
ResNet1D encoder receptive field analysis (Step 0 of concept-mixing check).

Answers: how much input signal does ONE feature-map position actually see?
This determines the temporal granularity at which concepts gather evidence.

Two methods:
  1. Theoretical RF: layer-by-layer recurrence
  2. Empirical impulse test: feed a delta signal, see which feature
     positions respond (eval mode, BN frozen).

Usage:
    python scripts/receptive_field.py
"""

import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.resnet1d import ResNet1D


def theoretical_receptive_field(model):
    """Layer-by-layer RF computation. Returns (rf, stride_cum, n_layers) per layer."""
    layers = [
        ("stem.conv", 7, 2),
        ("stem.maxpool", 3, 2),
    ]
    for li in range(1, 5):
        layer = getattr(model, f"layer{li}")
        for bi, block in enumerate(layer):
            s1 = block.conv1.stride[0]
            # NOTE: shortcut (1x1 conv) runs in PARALLEL with conv1, not in
            # sequence — it never extends the RF, and its stride does not
            # multiply stride_cum again (block output stride is already s1).
            layers.append((f"layer{li}.block{bi}.conv1", block.conv1.kernel_size[0], s1))
            layers.append((f"layer{li}.block{bi}.conv2", 3, 1))

    rows = []
    rf = 1
    stride_cum = 1
    for name, k, s in layers:
        rf = rf + (k - 1) * stride_cum
        stride_cum *= s
        rows.append((name, rf, stride_cum))
    return rows, rf, stride_cum


def empirical_impulse_test(model, seq_len=1000, device="cpu"):
    """
    Feed a delta at each of a set of input positions, record responding
    feature positions. Returns list of (input_pos, responding_feature_positions).
    """
    model.eval()
    # encoder only: up to layer4 (drop pool+fc by calling forward and grabbing features)
    results = []
    with torch.no_grad():
        for p in [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]:
            x = torch.zeros(1, 12, seq_len)
            x[:, :, p] = 1.0  # impulse on all 12 leads
            x = x.to(device)
            features, _ = model(x, return_features=True)  # [1, 512, L]
            # which feature positions have ANY nonzero response
            resp = (features.abs().sum(dim=1) > 0).squeeze(0).nonzero().flatten().tolist()
            results.append((p, resp))
    return results


def main():
    model = ResNet1D(in_channels=12, num_classes=5)
    rows, rf, stride = theoretical_receptive_field(model)

    print("=" * 70)
    print("Step 0: Encoder Receptive Field")
    print("=" * 70)
    print(f"\nTheoretical RF (layer-by-layer):")
    print(f"{'layer':<28s} {'RF (input samples)':>20s} {'cum stride':>12s}")
    print("-" * 62)
    for name, r, s in rows:
        print(f"{name:<28s} {r:>18d} {s:>12d}")
    print("-" * 62)
    print(f"FINAL theoretical RF: {rf} samples = {rf/100:.2f} s @ 100 Hz")
    print(f"Total downsampling: {stride}x")
    print(f"Feature map length for 1000-sample input: {1000//stride}")

    # coverage interval per feature position
    L = 1000 // stride
    print(f"\nEach feature position p covers input interval (theoretical):")
    print(f"  [p*{stride} - {rf//2}, p*{stride} + {rf//2}] clamped to [0, 1000]")
    print(f"  → window length {rf} samples ({rf/100:.2f} s)")
    print(f"  → adjacent positions spaced {stride} apart")
    print(f"  → overlap between adjacent windows: {(rf-stride)/rf*100:.1f}%")

    # empirical check on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nEmpirical impulse test (device={device}):")
    model = model.to(device)
    results = empirical_impulse_test(model, device=device)
    print(f"{'impulse pos':>12s} {'min feat pos':>13s} {'max feat pos':>13s} {'# responding':>12s}")
    print("-" * 55)
    for p, resp in results:
        if resp:
            print(f"{p:>12d} {min(resp):>13d} {max(resp):>13d} {len(resp):>12d}")
        else:
            print(f"{p:>12d} {'—':>13s} {'—':>13s} {'0 (none)':>12s}")

    # summarize empirical result: an impulse at input pos p is seen by every
    # output position whose RF window contains p → #responding ≈ RF/stride.
    # So the EMPIRICAL RF estimate = (#responding positions) × stride,
    # NOT the span between first and last responding position.
    center = [r for r in results if r[0] == 500][0]
    if center[1]:
        n_resp = len(center[1])
        emp_rf = n_resp * stride
        print(f"\nEmpirical: impulse at input pos 500 activates {n_resp} feature positions "
              f"({center[1][0]}..{center[1][-1]})")
        print(f"  → empirical RF estimate = {n_resp} × {stride} = {emp_rf} samples "
              f"({emp_rf/100:.2f} s)")
        print(f"  (each output position whose window contains the impulse responds)")


if __name__ == "__main__":
    main()
