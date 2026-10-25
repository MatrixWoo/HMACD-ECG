import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    """1D ResNet basic block with optional stride and channel change."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # shortcut: 1x1 conv when channels change or stride != 1
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for ECG classification.

    Input:  [B, in_channels, time_steps]   e.g. [B, 12, 1000]
    Output: [B, num_classes]               e.g. [B, 5]  (logits, no sigmoid)
    """

    def __init__(self, in_channels=12, num_classes=5):
        super().__init__()

        # ---- stem ----
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # ---- residual layers ----
        self.layer1 = self._make_layer(64, 64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # ---- head ----
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

        # ---- init ----
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock1d(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1d(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # feature map facts per layer (10s @100Hz input): channels, stride, RF in
    # input samples — see scripts/receptive_field.py for the RF derivation.
    LAYER_FACTS = {
        "layer1": {"channels": 64,  "stride": 4,  "rf": 43},
        "layer2": {"channels": 128, "stride": 8,  "rf": 99},
        "layer3": {"channels": 256, "stride": 16, "rf": 211},
        "layer4": {"channels": 512, "stride": 32, "rf": 435},
    }

    def forward(self, x, return_features=False, attach_layer="layer4"):
        """
        Args:
            x: [B, in_channels, time_steps]
            return_features: if True, also return spatial features before pool+fc
            attach_layer: which layer's feature map to return when
                return_features=True ("layer1".."layer4"). The pool+fc head
                always runs on layer4; the concept bank may attach earlier.
        Returns:
            if return_features=False:
                logits: [B, num_classes]
            if return_features=True:
                (features, logits)
                  features: [B, C_layer, L_layer]  — for the attached layer
                  logits:   [B, num_classes]       — layer4 head (unused by HMACD)
        """
        feats = {}
        out = self.stem(x)
        out = self.layer1(out)
        feats["layer1"] = out
        out = self.layer2(out)
        feats["layer2"] = out
        out = self.layer3(out)
        feats["layer3"] = out
        out = self.layer4(out)
        feats["layer4"] = out

        head_in = self.pool(out)          # [B, 512, 1]
        head_in = head_in.squeeze(-1)     # [B, 512]
        logits = self.fc(head_in)         # [B, num_classes]

        if return_features:
            if attach_layer not in feats:
                raise ValueError(f"unknown attach_layer {attach_layer}; "
                                 f"expected one of {list(feats)}")
            return feats[attach_layer], logits
        return logits


if __name__ == "__main__":
    model = ResNet1D(in_channels=12, num_classes=5)
    print(model)
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test 1: standard input [B, 12, 1000]
    x = torch.randn(4, 12, 1000)
    logits = model(x)
    assert logits.shape == (4, 5), f"Expected (4,5), got {logits.shape}"
    print(f"✅ Test 1 passed: input {x.shape} -> output {logits.shape}")

    # Test 2: variable-length input [B, 12, 2000]
    x_long = torch.randn(2, 12, 2000)
    logits_long = model(x_long)
    assert logits_long.shape == (2, 5), f"Expected (2,5), got {logits_long.shape}"
    print(f"✅ Test 2 passed: input {x_long.shape} -> output {logits_long.shape}")

    # Test 3: single sample
    x_single = torch.randn(1, 12, 1000)
    model.eval()
    with torch.no_grad():
        out = model(x_single)
    assert out.shape == (1, 5)
    print(f"✅ Test 3 passed: eval mode, single sample")

    # Test 4: gradient flow
    model.train()
    x.requires_grad = False
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"⚠️  No grad for {name}")
    print("✅ Test 4 passed: gradients flow")
