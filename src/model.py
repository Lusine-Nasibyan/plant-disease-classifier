"""
model.py

Defines the plant disease classifier built on top of a pretrained
backbone loaded via timm (PyTorch Image Models).

Supported backbones:
  - "efficientnet_b0"       : primary model (~5.3M params)
  - "mobilenetv3_small"     : ablation      (~2.5M params)

Two-stage fine-tuning strategy:
  Stage 1 — backbone frozen, only head trains  (fast convergence)
  Stage 2 — full model unfrozen, low LR        (careful refinement)

Usage:
    from src.model import build_model, freeze_backbone, unfreeze_backbone
"""

import timm
import torch
import torch.nn as nn
from pathlib import Path

# ─── SUPPORTED BACKBONES ──────────────────────────────────────────────────────

BACKBONES = {
    "efficientnet_b0":   "efficientnet_b0",
    "mobilenetv3_small": "mobilenetv3_small_100",
}


# ─── MODEL CLASS ──────────────────────────────────────────────────────────────

class DiseaseClassifier(nn.Module):
    """
    Wraps a timm feature extractor and a custom classification head.

    Having an explicit nn.Module with its own forward() method means
    the head is always called — there is no ambiguity about whether
    timm's internal forward routing picks it up or not.

    Attributes:
        backbone : timm feature extractor (outputs feature vectors)
        head     : Dropout + Linear mapping features to class logits
    """

    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone

        # Infer feature dimension with a single dummy pass
        with torch.no_grad():
            dummy      = torch.zeros(1, 3, 224, 224)
            feature_dim = self.backbone(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # [B, feature_dim]
        logits   = self.head(features) # [B, num_classes]
        return logits


# ─── MODEL BUILDER ────────────────────────────────────────────────────────────

def build_model(
    num_classes: int,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.3,
) -> DiseaseClassifier:
    """
    Build and return a DiseaseClassifier.

    Args:
        num_classes : number of disease classes (39 in your dataset)
        backbone    : key from BACKBONES dict
        pretrained  : load ImageNet weights
        dropout     : dropout rate in the classification head

    Why num_classes=0 in timm?
        Passing num_classes=0 tells timm to return a pure feature
        extractor — no classifier head attached. We then build and
        attach our own head inside DiseaseClassifier, where the
        forward() call is explicit and guaranteed.
    """
    backbone_name = BACKBONES.get(backbone)
    if backbone_name is None:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Choose from: {list(BACKBONES.keys())}"
        )

    feature_extractor = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=0,       # strip classifier — we attach our own
        global_pool="avg",   # global average pool before feature vector
    )

    return DiseaseClassifier(
        backbone=feature_extractor,
        num_classes=num_classes,
        dropout=dropout,
    )


# ─── FREEZE / UNFREEZE ────────────────────────────────────────────────────────

def freeze_backbone(model: DiseaseClassifier) -> None:
    """
    Freeze backbone, leave head trainable.

    Stage 1 of two-stage fine-tuning. The randomly initialized head
    trains first to produce stable gradients before we touch the
    pretrained backbone weights.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Backbone frozen   : {trainable:,} / {total:,} params trainable")


def unfreeze_backbone(model: DiseaseClassifier) -> None:
    """
    Unfreeze all parameters for Stage 2 fine-tuning.

    The backbone now adapts to plant disease features, but at a
    much lower LR than the head to avoid overwriting pretrained weights.
    """
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    print(f"  Backbone unfrozen : {total:,} / {total:,} params trainable")


# ─── PARAM GROUPS FOR DIFFERENTIAL LR ────────────────────────────────────────

def get_param_groups(
    model: DiseaseClassifier,
    head_lr: float,
    backbone_lr: float,
) -> list:
    """
    Two parameter groups with different learning rates for Stage 2.

    Differential LRs prevent the backbone from being updated too
    aggressively while still allowing it to adapt to disease features.

    Args:
        head_lr     : LR for the classification head
        backbone_lr : LR for the backbone (typically head_lr / 10)
    """
    return [
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.head.parameters(),     "lr": head_lr},
    ]


# ─── MODEL SUMMARY ────────────────────────────────────────────────────────────

def model_summary(model: DiseaseClassifier, backbone: str) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Backbone      : {backbone}")
    print(f"  Total params  : {total:,}")
    print(f"  Trainable     : {trainable:,}")
    print(f"  Head          : {model.head}")


# ─── SAVE / LOAD ──────────────────────────────────────────────────────────────

def save_model(
    model: DiseaseClassifier,
    path: Path,
    metadata: dict = None,
) -> None:
    """Save state_dict + optional metadata (epoch, val accuracy, etc.)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    if metadata:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, path)
    print(f"  Model saved → {path}")


def load_model(
    path: Path,
    num_classes: int,
    backbone: str = "efficientnet_b0",
) -> DiseaseClassifier:
    """Rebuild architecture then load saved weights."""
    model      = build_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    print(f"  Model loaded ← {path}")
    return model


# ─── SANITY CHECK ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NUM_CLASSES = 39

    for backbone_key in BACKBONES:
        print(f"\n{'='*50}")
        print(f"  Testing: {backbone_key}")
        print(f"{'='*50}")

        model = build_model(num_classes=NUM_CLASSES, backbone=backbone_key)
        model_summary(model, backbone_key)

        # Stage 1
        print("\n  --- Stage 1 (head only) ---")
        freeze_backbone(model)
        dummy = torch.zeros(4, 3, 224, 224)
        out   = model(dummy)
        print(f"  Output shape : {out.shape}")
        assert out.shape == (4, NUM_CLASSES), "Output shape mismatch!"
        print("  Forward pass : OK")

        # Stage 2
        print("\n  --- Stage 2 (full model) ---")
        unfreeze_backbone(model)
        groups = get_param_groups(model, head_lr=1e-3, backbone_lr=1e-4)
        print(f"  Param groups : {len(groups)}")
        print(f"  Backbone LR  : {groups[0]['lr']}")
        print(f"  Head LR      : {groups[1]['lr']}")

    print("\n  All checks passed.")