"""
train.py

Training script for plant disease classification.
Implements two-stage fine-tuning with weighted cross-entropy loss
and Weights & Biases experiment tracking.

Stages:
  Stage 1 : backbone frozen, head trains for `stage1_epochs`
  Stage 2 : full model unfrozen, differential LRs, trains for `stage2_epochs`

Run from project root:
    python src/train.py                          # default (EfficientNet, weighted CE)
    python src/train.py --backbone mobilenetv3_small          # ablation: backbone
    python src/train.py --no-weighted-loss                    # ablation: plain CE
    python src/train.py --use-weighted-sampler                # ablation: sampling
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from dataset import get_dataloaders, get_class_weights
from model   import (
    build_model, freeze_backbone, unfreeze_backbone,
    get_param_groups, save_model,
)

# ─── CONFIG DEFAULTS ──────────────────────────────────────────────────────────

DEFAULT_CFG = {
    # Data
    "labels_csv"           : "data/processed/labels.csv",
    "num_classes"          : 39,

    # Model
    "backbone"             : "efficientnet_b0",
    "dropout"              : 0.3,

    # Training stages
    "stage1_epochs"        : 5,     # head-only warmup
    "stage2_epochs"        : 20,    # full fine-tune

    # Learning rates
    "head_lr"              : 1e-3,  # head LR in both stages
    "backbone_lr"          : 1e-4,  # backbone LR in stage 2 (10× lower)

    # Optimiser
    "weight_decay"         : 1e-4,

    # Batch
    "batch_size"           : 32,

    # Loss / sampling strategy (ablation flags)
    "weighted_loss"        : True,
    "use_weighted_sampler" : False,

    # Saving
    "save_dir"             : "models",
    "save_best_metric"     : "val_acc",   # checkpoint on best val accuracy

    # Reproducibility
    "seed"                 : 42,
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Fix all random seeds for reproducibility.
    Without this, two runs with the same config can give different results
    due to random weight init and data shuffling order.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """
    Use GPU if available, otherwise CPU.
    On Colab this will pick up the T4/A100 automatically.
    Locally on Windows without a GPU this falls back to CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    return device


def run_name(cfg: dict) -> str:
    """
    Construct a human-readable W&B run name from the config.
    Makes it easy to distinguish runs in the W&B dashboard.

    Example: "efficientnet_b0__wloss__s1-5_s2-20"
    """
    loss_tag    = "wloss"  if cfg["weighted_loss"]        else "plainCE"
    sampler_tag = "_wsamp" if cfg["use_weighted_sampler"] else ""
    return (
        f"{cfg['backbone']}__{loss_tag}{sampler_tag}"
        f"__s1-{cfg['stage1_epochs']}_s2-{cfg['stage2_epochs']}"
    )


# ─── TRAIN ONE EPOCH ──────────────────────────────────────────────────────────
from tqdm import tqdm

def train_epoch(
    model      : nn.Module,
    loader     : torch.utils.data.DataLoader,
    criterion  : nn.Module,
    optimiser  : torch.optim.Optimizer,
    device     : torch.device,
    stage      : int,
) -> tuple[float, float]:
    """
    Run one full pass over the training set.

    Returns:
        avg_loss : mean loss over all batches
        accuracy : fraction of correct predictions
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc=f"S{stage} train", leave=False, unit="batch")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — caps gradient norm at 1.0 to prevent
        # large gradient spikes from destabilising training,
        # especially important in stage 2 when the full model is unfrozen.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


# ─── VALIDATE ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model     : nn.Module,
    loader    : torch.utils.data.DataLoader,
    criterion : nn.Module,
    device    : torch.device,
) -> tuple[float, float]:
    """
    Evaluate on the validation set.

    @torch.no_grad() disables gradient tracking entirely during
    validation — saves memory and speeds up the pass since we
    never call .backward() here.

    Returns:
        avg_loss : mean val loss
        accuracy : fraction of correct predictions
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    """
    Full two-stage training loop.

    Stage 1 — frozen backbone:
        Only the head trains. We use a single LR (head_lr) and
        a short cosine schedule. This warms up the head quickly
        before we risk touching pretrained backbone weights.

    Stage 2 — full model:
        Both backbone and head train with differential LRs.
        The backbone gets backbone_lr (10× smaller than head_lr)
        to update carefully. Cosine annealing decays both LRs
        smoothly to near-zero by the end of training.

    Why AdamW?
        Adam with decoupled weight decay. Decoupled weight decay
        (AdamW) is more principled than L2 regularisation inside
        Adam and consistently outperforms plain Adam on fine-tuning
        tasks. It is the standard optimiser for transfer learning.

    Why CosineAnnealingLR?
        Cosine annealing decays the LR along a smooth cosine curve
        to near-zero. Compared to step decay it avoids abrupt LR
        drops and tends to find flatter minima that generalise better.
    """
    set_seed(cfg["seed"])
    device = get_device()

    # ── W&B init ──────────────────────────────────────────────────────────
    # All hyperparameters are logged so every run is fully reproducible
    # from the W&B dashboard alone.
    wandb.init(
        project = "plant-disease-classifier",
        name    = run_name(cfg),
        config  = cfg,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    train_loader, val_loader, train_dataset = get_dataloaders(
        labels_csv           = Path(cfg["labels_csv"]),
        batch_size           = cfg["batch_size"],
        use_weighted_sampler = cfg["use_weighted_sampler"],
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n=== Building model ===")
    model = build_model(
        num_classes = cfg["num_classes"],
        backbone    = cfg["backbone"],
        dropout     = cfg["dropout"],
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────
    # Weighted CE: each class gets a weight inversely proportional to its
    # frequency. Rare classes contribute more to the loss, forcing the model
    # to take them seriously despite seeing fewer examples.
    # Plain CE (ablation): all classes treated equally regardless of count.
    if cfg["weighted_loss"]:
        class_weights = get_class_weights(train_dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("  Loss : Weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("  Loss : Plain CrossEntropyLoss")

    # ── Save dir ──────────────────────────────────────────────────────────
    save_dir  = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_metric = 0.0
    best_path   = save_dir / f"{run_name(cfg)}__best.pt"

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1 — Head only
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n=== Stage 1: head warmup ({cfg['stage1_epochs']} epochs) ===")
    freeze_backbone(model)

    optimiser = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg["head_lr"],
        weight_decay = cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimiser,
        T_max = cfg["stage1_epochs"],
        eta_min = cfg["head_lr"] * 0.01,  # decay to 1% of initial LR
    )

    for epoch in range(1, cfg["stage1_epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimiser, device, stage=1
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Log to W&B
        wandb.log({
            "stage"      : 1,
            "epoch"      : epoch,
            "train/loss" : train_loss,
            "train/acc"  : train_acc,
            "val/loss"   : val_loss,
            "val/acc"    : val_acc,
            "lr"         : optimiser.param_groups[0]["lr"],
        })

        print(
            f"  [S1 E{epoch:02d}] "
            f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.4f}"
        )

        # Save best checkpoint
        if val_acc > best_metric:
            best_metric = val_acc
            save_model(model, best_path, metadata={"epoch": epoch, "val_acc": val_acc})
            wandb.summary["best_val_acc"] = best_metric

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2 — Full model, differential LRs
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n=== Stage 2: full fine-tune ({cfg['stage2_epochs']} epochs) ===")
    unfreeze_backbone(model)

    # Differential LRs: backbone gets 10× lower LR than head.
    # We re-create the optimiser because the set of trainable parameters
    # has changed — stage 1 optimiser only knew about head params.
    param_groups = get_param_groups(
        model,
        head_lr     = cfg["head_lr"],
        backbone_lr = cfg["backbone_lr"],
    )
    optimiser = AdamW(param_groups, weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(
        optimiser,
        T_max   = cfg["stage2_epochs"],
        eta_min = cfg["backbone_lr"] * 0.01,
    )

    for epoch in range(1, cfg["stage2_epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimiser, device, stage=2
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Log both LRs separately so W&B shows backbone vs head LR curves
        wandb.log({
            "stage"        : 2,
            "epoch"        : cfg["stage1_epochs"] + epoch,
            "train/loss"   : train_loss,
            "train/acc"    : train_acc,
            "val/loss"     : val_loss,
            "val/acc"      : val_acc,
            "lr/backbone"  : optimiser.param_groups[0]["lr"],
            "lr/head"      : optimiser.param_groups[1]["lr"],
        })

        print(
            f"  [S2 E{epoch:02d}] "
            f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.4f}"
        )

        if val_acc > best_metric:
            best_metric = val_acc
            save_model(model, best_path, metadata={"epoch": epoch, "val_acc": val_acc})
            wandb.summary["best_val_acc"] = best_metric

    # ── Final save & W&B finish ───────────────────────────────────────────
    final_path = save_dir / f"{run_name(cfg)}__final.pt"
    save_model(model, final_path, metadata={"val_acc": val_acc})

    # Save class mapping alongside weights so inference always has it
    class_map_path = save_dir / "class_mapping.json"
    with open(class_map_path, "w") as f:
        json.dump(train_dataset.idx_to_class, f, indent=2)
    print(f"  Class map saved → {class_map_path}")

    wandb.finish()
    print(f"\n  Best val acc : {best_metric:.4f}")
    print(f"  Best model   : {best_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> dict:
    """
    Parse command-line arguments and merge with DEFAULT_CFG.
    CLI args override defaults — makes ablation runs one-liners.
    """
    parser = argparse.ArgumentParser(description="Train plant disease classifier")

    parser.add_argument("--backbone",    default=DEFAULT_CFG["backbone"],
                        choices=["efficientnet_b0", "mobilenetv3_small"])
    parser.add_argument("--stage1-epochs", type=int, default=DEFAULT_CFG["stage1_epochs"])
    parser.add_argument("--stage2-epochs", type=int, default=DEFAULT_CFG["stage2_epochs"])
    parser.add_argument("--batch-size",    type=int, default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--head-lr",       type=float, default=DEFAULT_CFG["head_lr"])
    parser.add_argument("--backbone-lr",   type=float, default=DEFAULT_CFG["backbone_lr"])
    parser.add_argument("--dropout",       type=float, default=DEFAULT_CFG["dropout"])
    parser.add_argument("--no-weighted-loss",      dest="weighted_loss",
                        action="store_false",
                        help="Ablation: use plain cross-entropy instead of weighted")
    parser.add_argument("--use-weighted-sampler",  dest="use_weighted_sampler",
                        action="store_true",
                        help="Ablation: use WeightedRandomSampler")
    parser.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])

    args = parser.parse_args()
    cfg  = DEFAULT_CFG.copy()
    cfg.update({
        "backbone"             : args.backbone,
        "stage1_epochs"        : args.stage1_epochs,
        "stage2_epochs"        : args.stage2_epochs,
        "batch_size"           : args.batch_size,
        "head_lr"              : args.head_lr,
        "backbone_lr"          : args.backbone_lr,
        "dropout"              : args.dropout,
        "weighted_loss"        : args.weighted_loss,
        "use_weighted_sampler" : args.use_weighted_sampler,
        "seed"                 : args.seed,
    })
    return cfg


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = parse_args()

    print("=== Run configuration ===")
    for k, v in cfg.items():
        print(f"  {k:<25} {v}")

    train(cfg)