"""
evaluate.py

Computes per-class Average Precision (AP) and mean Average Precision (mAP)
on the validation set. These are the primary metrics for this project.

Why mAP and not just accuracy?
    Accuracy treats all classes equally by count — on an imbalanced dataset
    it can be high even if the model completely ignores rare classes.
    AP measures the area under the precision-recall curve per class,
    which evaluates quality across the full confidence range, not just
    at a single threshold. mAP averages this across all classes, giving
    every disease equal weight regardless of how many images it has.

Run from project root:
    python src/evaluate.py --checkpoint models/<run_name>__best.pt
    python src/evaluate.py --checkpoint models/<run_name>__best.pt --backbone mobilenetv3_small
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from dataset import get_dataloaders, PlantDiseaseDataset
from model   import load_model


# ─── INFERENCE PASS ───────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model  : torch.nn.Module,
    loader : torch.utils.data.DataLoader,
    device : torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over the entire val set and collect:
        all_probs  : softmax probabilities  [N, num_classes]
        all_preds  : argmax predictions     [N]
        all_labels : ground truth indices   [N]

    We collect probabilities (not just argmax predictions) because
    AP is computed from the full probability distribution, not just
    the top-1 prediction. This gives a much richer signal — a model
    that is confidently wrong is penalised more than one that is
    uncertain but wrong.
    """
    model.eval()
    all_probs  = []
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)

        # Convert logits to probabilities with softmax
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_probs,  axis=0),
        np.concatenate(all_preds,  axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ─── mAP COMPUTATION ──────────────────────────────────────────────────────────

def compute_map(
    probs      : np.ndarray,
    labels     : np.ndarray,
    num_classes: int,
) -> tuple[float, dict[int, float]]:
    """
    Compute per-class AP and macro-averaged mAP.

    AP for class c is computed as:
        - Binarise labels: 1 if label == c, else 0
        - Rank all samples by their predicted probability for class c
        - Compute area under the precision-recall curve

    Macro average (unweighted mean across classes) is used because
    we want every disease class to contribute equally to the final
    score, regardless of how many val images it has.

    Args:
        probs       : softmax probabilities [N, num_classes]
        labels      : ground truth class indices [N]
        num_classes : total number of classes

    Returns:
        map_score   : scalar mean AP across all classes
        per_class_ap: dict mapping class index -> AP score
    """
    # One-hot encode ground truth for sklearn
    # Shape: [N, num_classes]
    labels_onehot = np.zeros((len(labels), num_classes), dtype=int)
    labels_onehot[np.arange(len(labels)), labels] = 1

    per_class_ap = {}
    for c in range(num_classes):
        # Skip classes with no positive samples in val
        # (shouldn't happen with your dataset but defensive)
        if labels_onehot[:, c].sum() == 0:
            continue
        ap = average_precision_score(
            labels_onehot[:, c],
            probs[:, c],
        )
        per_class_ap[c] = ap

    map_score = float(np.mean(list(per_class_ap.values())))
    return map_score, per_class_ap


# ─── RESULTS PRINTING ─────────────────────────────────────────────────────────

def print_results(
    map_score    : float,
    per_class_ap : dict[int, float],
    idx_to_class : dict[int, str],
    all_preds    : np.ndarray,
    all_labels   : np.ndarray,
    top_k        : int = 5,
) -> None:
    """
    Print a full evaluation report:
      1. mAP score
      2. Per-class AP table sorted by AP (worst to best)
      3. Top-K and bottom-K classes
      4. Top-1 accuracy
      5. Per-class precision, recall, F1
    """
    num_classes = len(per_class_ap)

    # ── mAP ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  mAP (macro, {num_classes} classes) : {map_score:.4f}  ({map_score*100:.2f}%)")
    print(f"{'='*65}")

    # ── Per-class AP table ─────────────────────────────────────────────────
    sorted_by_ap = sorted(per_class_ap.items(), key=lambda x: x[1])

    print(f"\n{'Disease':<48} {'AP':>6}")
    print("─" * 56)
    for class_idx, ap in sorted_by_ap:
        name = idx_to_class.get(class_idx, str(class_idx))
        bar  = "█" * int(ap * 20)   # simple visual bar (max 20 chars)
        print(f"  {name:<46} {ap:.4f}  {bar}")
    print("─" * 56)

    # ── Best and worst classes ────────────────────────────────────────────
    print(f"\n  Bottom {top_k} (hardest to classify):")
    for class_idx, ap in sorted_by_ap[:top_k]:
        name = idx_to_class.get(class_idx, str(class_idx))
        print(f"    {name:<44}  AP = {ap:.4f}")

    print(f"\n  Top {top_k} (easiest to classify):")
    for class_idx, ap in sorted_by_ap[-top_k:]:
        name = idx_to_class.get(class_idx, str(class_idx))
        print(f"    {name:<44}  AP = {ap:.4f}")

    # ── Top-1 accuracy ────────────────────────────────────────────────────
    top1_acc = (all_preds == all_labels).mean()
    print(f"\n  Top-1 Accuracy : {top1_acc:.4f}  ({top1_acc*100:.2f}%)")

    # ── Per-class precision / recall / F1 ─────────────────────────────────
    class_names = [idx_to_class[i] for i in range(num_classes)]
    print("\n  Per-class Classification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )


# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────

def save_results(
    map_score    : float,
    per_class_ap : dict[int, float],
    idx_to_class : dict[int, str],
    out_path     : Path,
) -> None:
    """
    Save evaluation results to JSON for later comparison across runs.
    Having results in a structured file makes it easy to build comparison
    tables in your report without re-running evaluation.
    """
    results = {
        "mAP"          : map_score,
        "per_class_ap" : {
            idx_to_class[k]: round(v, 6)
            for k, v in per_class_ap.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def evaluate(cfg: dict) -> float:
    """
    Full evaluation pipeline. Returns mAP for programmatic use
    (e.g. calling from a Colab notebook after training).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    # ── Load class mapping ────────────────────────────────────────────────
    # We load idx_to_class from the JSON saved during training rather than
    # re-deriving it, ensuring evaluation uses the exact same mapping.
    class_map_path = Path(cfg["class_map"])
    with open(class_map_path) as f:
        raw = json.load(f)
    # JSON keys are always strings — convert back to int
    idx_to_class = {int(k): v for k, v in raw.items()}
    num_classes  = len(idx_to_class)
    print(f"  Classes : {num_classes}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n=== Loading val data ===")
    _, val_loader, _ = get_dataloaders(
        batch_size=cfg["batch_size"],
        use_weighted_sampler=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n=== Loading model ===")
    model = load_model(
        path        = Path(cfg["checkpoint"]),
        num_classes = num_classes,
        backbone    = cfg["backbone"],
    ).to(device)

    # ── Inference ─────────────────────────────────────────────────────────
    print("\n=== Running inference on val set ===")
    all_probs, all_preds, all_labels = get_predictions(model, val_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────
    print("\n=== Computing metrics ===")
    map_score, per_class_ap = compute_map(all_probs, all_labels, num_classes)
    print_results(map_score, per_class_ap, idx_to_class, all_preds, all_labels)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(cfg["out_dir"]) / f"{Path(cfg['checkpoint']).stem}__results.json"
    save_results(map_score, per_class_ap, idx_to_class, out_path)

    return map_score


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Evaluate plant disease classifier")

    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt checkpoint file, e.g. models/efficientnet_b0__wloss__best.pt"
    )
    parser.add_argument(
        "--backbone", default="efficientnet_b0",
        choices=["efficientnet_b0", "mobilenetv3_small"],
    )
    parser.add_argument(
        "--class-map", default="models/class_mapping.json",
        help="Path to class_mapping.json saved during training"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir",    default="models",
                        help="Directory to save results JSON")

    args = parser.parse_args()
    return {
        "checkpoint" : args.checkpoint,
        "backbone"   : args.backbone,
        "class_map"  : args.class_map,
        "batch_size" : args.batch_size,
        "out_dir"    : args.out_dir,
    }


if __name__ == "__main__":
    cfg = parse_args()
    print("=== Evaluation config ===")
    for k, v in cfg.items():
        print(f"  {k:<20} {v}")
    evaluate(cfg)