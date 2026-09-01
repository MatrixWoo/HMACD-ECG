"""
Model-Discovered Concept Bank for HMACD-ECG.

Takes spatial features from ECG encoder backbone and computes
concept activations via learnable prototypes:

    z_k = max_{l} cosine_sim(h_l, p_k)

where h_l is the feature vector at spatial position l,
and p_k is the k-th learnable prototype.

Supports:
- concept intervention (zero out specific concepts)
- compactness loss (encourage localized activations)
- diversity loss (encourage distinct prototypes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptBank(nn.Module):
    """
    Model-discovered concept bank with K learnable prototypes.

    Args:
        in_channels:   feature dimension C from backbone (e.g. 512)
        num_concepts:  K — number of concept prototypes (default 32)
        num_classes:   number of output classes (default 5)
    """

    def __init__(self, in_channels=512, num_concepts=32, num_classes=5):
        super().__init__()
        self.in_channels = in_channels
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        # ---- learnable prototypes ----
        self.prototypes = nn.Parameter(
            torch.randn(num_concepts, in_channels) * 0.01
        )

        # ---- concept → class classifier ----
        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, features):
        """
        Args:
            features: [B, C, L]  spatial feature map from backbone (e.g. [B, 512, 32])
        Returns:
            logits:        [B, num_classes]  classification logits
            z:             [B, K]            concept activations
            sim:           [B, K, L]         similarity map (for visualization)
        """
        B, C, L = features.shape

        # ---- cosine similarity between features and prototypes ----
        # normalize along channel dim
        h_norm = F.normalize(features, p=2, dim=1)     # [B, C, L]
        p_norm = F.normalize(self.prototypes, p=2, dim=1)  # [K, C]

        # similarity map: for each (batch, prototype, position)
        sim = torch.einsum('bcl,kc->bkl', h_norm, p_norm)  # [B, K, L]

        # ---- z_k = max over spatial positions ----
        z, _ = sim.max(dim=-1)  # [B, K]

        # ---- classifier ----
        logits = self.classifier(z)  # [B, num_classes]

        return logits, z, sim

    @torch.no_grad()
    def get_top_activations(self, features, top_k=5):
        """
        Return top activating positions for visualization.

        Args:
            features: [1, C, L]  single-sample feature map
            top_k:    number of top segments per concept
        Returns:
            top_positions: [K, top_k]  — time indices of top activations
            top_scores:    [K, top_k]  — corresponding similarity scores
        """
        self.eval()
        _, _, sim = self.forward(features)      # [1, K, L]
        sim = sim.squeeze(0)                     # [K, L]
        top_scores, top_positions = sim.topk(top_k, dim=-1)  # [K, top_k]
        return top_positions, top_scores

    def intervene(self, z, concept_indices):
        """
        Suppress specific concepts (set activations to zero).

        Args:
            z:               [B, K]  original concept activations
            concept_indices: list of int — which concepts to suppress
        Returns:
            z_masked: [B, K]  modified activations
        """
        z_masked = z.clone()
        z_masked[:, concept_indices] = 0.0
        return z_masked

    def classify_from_z(self, z):
        """Get logits from concept activations (used with intervention)."""
        return self.classifier(z)


def concept_losses(logits, y, sim, prototypes,
                   lambda_compact=0.1, lambda_diverse=0.05):
    """
    Compute concept-related auxiliary losses.

    Args:
        logits:          [B, num_classes]
        y:               [B, num_classes] ground-truth labels
        sim:             [B, K, L]          similarity map
        prototypes:      [K, C]             concept prototypes
        lambda_compact:  weight for compactness loss
        lambda_diverse:  weight for diversity loss

    Returns:
        total_loss:  scalar = cls_loss + λ_c*compactness + λ_d*diversity
        loss_dict:   dict with individual loss values
    """
    # ---- classification loss ----
    cls_loss = F.binary_cross_entropy_with_logits(logits, y)

    # ---- compactness loss: encourage localized concept activation ----
    attn = F.softmax(sim, dim=-1)                              # [B, K, L]
    entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1)     # [B, K]
    L = sim.shape[-1]
    entropy_max = torch.log(torch.tensor(L, dtype=sim.dtype, device=sim.device))
    compactness = entropy.mean() / entropy_max                  # 0=localized, 1=uniform

    # ---- diversity loss: encourage distinct prototypes ----
    diversity = compute_prototype_diversity(prototypes)

    total_loss = cls_loss + lambda_compact * compactness + lambda_diverse * diversity

    loss_dict = {
        "cls_loss": cls_loss.item(),
        "compactness": compactness.item(),
        "diversity": diversity.item(),
        "total": total_loss.item(),
    }
    return total_loss, loss_dict


def compute_prototype_diversity(prototypes):
    """
    Compute mean off-diagonal cosine similarity between prototypes.
    Lower value → more diverse prototypes.

    Args:
        prototypes: [K, C]
    Returns:
        diversity: scalar (0 = orthogonal, 1 = all identical)
    """
    p_norm = F.normalize(prototypes, p=2, dim=1)   # [K, C]
    proto_sim = p_norm @ p_norm.T                    # [K, K]
    K = proto_sim.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool, device=proto_sim.device)
    diversity = proto_sim[mask].mean()
    return diversity


class HMACDModel(nn.Module):
    """
    Full HMACD-ECG model: backbone + concept bank.

    Pipeline:
        Raw ECG [B, 12, T]
          → backbone (ResNet1D) → spatial features [B, C, L]
          → concept bank → z [B, K] + logits [B, num_classes]

    Supports concept intervention and per-concept analysis.
    """

    def __init__(self, backbone, num_concepts=32, num_classes=5):
        super().__init__()
        self.backbone = backbone
        self.concept_bank = ConceptBank(
            in_channels=512,  # matches ResNet1D layer4 output channels
            num_concepts=num_concepts,
            num_classes=num_classes,
        )

    def forward(self, x, intervene_concepts=None):
        """
        Args:
            x: [B, 12, T]
            intervene_concepts: optional list of concept indices to suppress
        Returns:
            logits: [B, num_classes]
            z:      [B, K]            concept activations
            sim:    [B, K, L]         similarity map
        """
        features, _ = self.backbone(x, return_features=True)
        logits, z, sim = self.concept_bank(features)

        if intervene_concepts is not None:
            z = self.concept_bank.intervene(z, intervene_concepts)
            logits = self.concept_bank.classify_from_z(z)

        return logits, z, sim

    def get_prototypes(self):
        """Return prototype vectors [K, C]."""
        return self.concept_bank.prototypes

    def compute_pcs(self, x, class_idx):
        """
        Predictive Contribution Score for a specific class.
        PCS_k = ŷ_c - ŷ_c^{(k)} where concept k is suppressed.

        Args:
            x:         [N, 12, T]
            class_idx: int — target class index
        Returns:
            pcs: [N, K]  per-sample per-concept contribution scores
        """
        self.eval()
        with torch.no_grad():
            logits_full, z_full, _ = self.forward(x)
            probs_full = torch.sigmoid(logits_full)[:, class_idx]  # [N]

            N, K = z_full.shape
            pcs = torch.zeros(N, K, device=x.device)
            for k in range(K):
                logits_k, _, _ = self.forward(x, intervene_concepts=[k])
                probs_k = torch.sigmoid(logits_k)[:, class_idx]
                pcs[:, k] = probs_full - probs_k

        return pcs


if __name__ == "__main__":
    from src.models.resnet1d import ResNet1D

    print("=" * 60)
    print("ConceptBank unit tests")
    print("=" * 60)

    # setup
    backbone = ResNet1D(in_channels=12, num_classes=5)
    concept_bank = ConceptBank(in_channels=512, num_concepts=32, num_classes=5)

    x = torch.randn(4, 12, 1000)
    features, _ = backbone(x, return_features=True)

    # Test 1: forward
    logits, z, sim = concept_bank(features)
    print(f"features: {features.shape}")
    print(f"logits: {logits.shape}   (expect [4, 5])")
    print(f"z: {z.shape}            (expect [4, 32])")
    print(f"sim: {sim.shape}       (expect [4, 32, 32])")
    assert logits.shape == (4, 5)
    assert z.shape == (4, 32)
    assert sim.shape == (4, 32, features.shape[-1])
    print("✅ Test 1 passed: shapes correct")

    # Test 2: concept intervention
    z_masked = concept_bank.intervene(z, [0, 5, 10])
    logits_masked = concept_bank.classify_from_z(z_masked)
    assert logits_masked.shape == (4, 5)
    assert (z_masked[:, 0] == 0).all()
    print("✅ Test 2 passed: concept intervention works")

    # Test 3: prototype diversity
    div = compute_prototype_diversity(concept_bank.prototypes)
    print(f"Initial diversity: {div:.4f} (random init, near 0 expected)")
    print("✅ Test 3 passed: diversity computed")

    # Test 4: top activations
    x1 = torch.randn(1, 12, 1000)
    f1, _ = backbone(x1, return_features=True)
    positions, scores = concept_bank.get_top_activations(f1, top_k=3)
    assert positions.shape == (32, 3)
    print(f"Top positions per concept: shape {positions.shape}")
    print(f"Top scores range: [{scores.min():.3f}, {scores.max():.3f}]")
    print("✅ Test 4 passed: top activations")

    # Test 5: gradient flow
    y = torch.randint(0, 2, (4, 5)).float()
    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    grads_ok = all(p.grad is not None for p in concept_bank.parameters())
    print(f"✅ Test 5: gradient flow {'OK' if grads_ok else 'FAIL'}")
