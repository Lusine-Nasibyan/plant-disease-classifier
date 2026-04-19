"""
dataset.py

PyTorch Dataset for plant disease classification.
Reads from data/processed/labels.csv and serves (image_tensor, label_index)
pairs to the training loop.

Also exposes:
  - get_class_weights()  : inverse-frequency weights for weighted CE loss
  - get_transforms()     : train (augmented) and val (clean) transform pipelines
  - get_dataloaders()    : returns train and val DataLoader objects

Usage:
    from src.dataset import get_dataloaders, get_class_weights
"""

import csv
from pathlib import Path
from collections import Counter
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# ─── CONFIG ───────────────────────────────────────────────────────────────────

LABELS_CSV = Path("data/processed/labels.csv")
IMAGE_SIZE = 224      # EfficientNet-B0 expects 224x224
NUM_WORKERS = 0       # Windows does not support multiprocessing workers
                      # in DataLoader well — keep at 0 to avoid errors

# ImageNet mean and std — used because we load ImageNet pretrained weights.
# Normalizing with these values puts our images in the same distribution
# the backbone was originally trained on.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── TRANSFORMS ───────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Return image transforms for a given split.

    Train transforms apply data augmentation to artificially expand
    the effective dataset size and improve generalization:
      - RandomResizedCrop: forces the model to handle partial views and scale
      - RandomHorizontalFlip: disease patterns are not directionally biased
      - ColorJitter: accounts for different lighting, cameras, leaf colors
      - RandomRotation: leaves are photographed at arbitrary angles
      - RandomGrayscale: teaches the model not to rely purely on color

    Val transforms apply NO augmentation — only the same resize and
    normalize needed for a clean, reproducible evaluation.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.RandomRotation(degrees=20),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ─── DATASET ──────────────────────────────────────────────────────────────────

class PlantDiseaseDataset(Dataset):
    """
    Reads labels.csv and serves (image_tensor, label_index) pairs.

    Args:
        labels_csv : path to data/processed/labels.csv
        split      : "train" or "val"
        transform  : torchvision transform pipeline

    The class-to-index mapping (class_to_idx) is built from the full
    CSV regardless of split, so train and val always share the same
    integer encoding. This is critical — if train encodes "rust" as 5
    and val encodes it as 12, every prediction is wrong.
    """

    def __init__(
        self,
        labels_csv: Path = LABELS_CSV,
        split: str = "train",
        transform=None,
    ):
        self.split     = split
        self.transform = transform
        self.samples   = []   # list of (image_path, label_index)

        # ── Build class index from ALL rows (both splits) ─────────────────
        # Sort ensures the mapping is deterministic — same order every run.
        all_diseases = set()
        rows_for_split = []

        with open(labels_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

        for row in all_rows:
            all_diseases.add(row["disease_label"].strip())

        self.classes     = sorted(all_diseases)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        # ── Load only rows matching this split ────────────────────────────
        for row in all_rows:
            if row["split"].strip() != split:
                continue
            img_path = Path(row["image_path"].strip())
            disease  = row["disease_label"].strip()
            label    = self.class_to_idx[disease]
            self.samples.append((img_path, label))

        print(
            f"  [{split}] {len(self.samples)} images | "
            f"{len(self.classes)} disease classes"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Load image — convert to RGB to handle any grayscale or RGBA images
        # that might be in the dataset without crashing.
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ─── CLASS WEIGHTS ────────────────────────────────────────────────────────────

def get_class_weights(dataset: PlantDiseaseDataset) -> torch.Tensor:
    """
    Compute inverse-frequency weights for weighted cross-entropy loss.

    Weight for class c = total_samples / (num_classes * count_of_c)

    This means rare classes receive a proportionally higher weight,
    forcing the loss function to penalize mistakes on them more heavily.
    The result is a 1D tensor of length num_classes, passed directly
    to nn.CrossEntropyLoss(weight=...).

    Example with your data:
      powdery mildew (866 samples) -> low weight  (~0.38)
      stem rust      (90 samples)  -> high weight (~3.6)
    """
    label_counts = Counter(label for _, label in dataset.samples)
    num_classes  = len(dataset.classes)
    total        = len(dataset.samples)

    weights = torch.zeros(num_classes)
    for class_idx, count in label_counts.items():
        weights[class_idx] = total / (num_classes * count)

    return weights


# ─── WEIGHTED SAMPLER (for ablations) ─────────────────────────────────────────

def get_weighted_sampler(dataset: PlantDiseaseDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler for oversampling minority classes.

    Each sample is assigned a weight equal to its class weight, then
    PyTorch samples with replacement according to those weights.
    This causes minority-class images to appear more frequently per epoch.

    NOTE: Used only in ablation experiments. Our primary strategy uses
    weighted loss with uniform sampling instead, since minority classes
    have few enough images (~90) that heavy oversampling risks overfitting.
    """
    class_weights = get_class_weights(dataset)
    sample_weights = torch.tensor(
        [class_weights[label] for _, label in dataset.samples]
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─── DATALOADERS ──────────────────────────────────────────────────────────────

def get_dataloaders(
    labels_csv: Path = LABELS_CSV,
    batch_size: int = 32,
    use_weighted_sampler: bool = False,   # set True for ablation
) -> tuple[DataLoader, DataLoader, PlantDiseaseDataset]:
    """
    Construct and return (train_loader, val_loader, train_dataset).

    train_dataset is returned alongside loaders so the training script
    can call get_class_weights(train_dataset) without re-reading the CSV.

    Args:
        labels_csv           : path to labels.csv
        batch_size           : images per batch
        use_weighted_sampler : if True, use WeightedRandomSampler instead
                               of uniform sampling (ablation mode)
    """
    train_dataset = PlantDiseaseDataset(
        labels_csv=labels_csv,
        split="train",
        transform=get_transforms("train"),
    )
    val_dataset = PlantDiseaseDataset(
        labels_csv=labels_csv,
        split="val",
        transform=get_transforms("val"),
    )

    # Sampler setup
    if use_weighted_sampler:
        sampler = get_weighted_sampler(train_dataset)
        shuffle = False      # shuffle and sampler are mutually exclusive in PyTorch
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,     # speeds up CPU→GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,        # never shuffle val — results must be reproducible
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset


# ─── QUICK SANITY CHECK ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Sanity check ===\n")

    train_loader, val_loader, train_ds = get_dataloaders(batch_size=32)

    # Check one batch loads correctly
    images, labels = next(iter(train_loader))
    print(f"  Batch image tensor : {images.shape}")   # expect [32, 3, 224, 224]
    print(f"  Batch label tensor : {labels.shape}")   # expect [32]
    print(f"  Label range        : {labels.min().item()} – {labels.max().item()}")
    print(f"  Pixel mean (approx): {images.mean():.4f}")  # should be near 0

    # Check class weights
    weights = get_class_weights(train_ds)
    print(f"\n  Class weight tensor shape : {weights.shape}")
    print(f"  Min weight : {weights.min():.4f}  (most common class)")
    print(f"  Max weight : {weights.max():.4f}  (rarest class)")

    # Print a few class → index mappings
    print("\n  Sample class mapping:")
    for cls in list(train_ds.class_to_idx.keys())[:5]:
        print(f"    '{cls}' → {train_ds.class_to_idx[cls]}")