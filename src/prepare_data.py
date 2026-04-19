"""
prepare_data.py

Builds a labelled CSV (data/processed/labels.csv) mapping every image
to its disease class, using the hand-verified class_mapping.csv as the
source of truth for labels.

Run once from the project root before any training:
    python src/prepare_data.py
"""

import csv
from pathlib import Path
from collections import Counter

# ─── CONFIG ───────────────────────────────────────────────────────────────────

RAW_TRAIN    = Path("data/raw/train")
RAW_VAL      = Path("data/raw/val")
CLASS_MAP    = Path("data/metadata/class_mapping.csv")
OUT_CSV      = Path("data/processed/labels.csv")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ─── STEP 1: LOAD CLASS MAPPING ───────────────────────────────────────────────

def load_class_mapping(csv_path: Path) -> dict[str, str]:
    """
    Read class_mapping.csv and return a dict of:
        { original_class -> disease }

    Example entry:
        "tomato late blight" -> "late blight"

    Using this file as the source of truth means the disease labels
    are explicit and auditable, not inferred by string manipulation.
    """
    mapping = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row["original_class"].strip().lower()
            disease  = row["disease"].strip().lower()
            mapping[original] = disease

    print(f"  Loaded {len(mapping)} class mappings from {csv_path}")
    return mapping

# ─── STEP 2: BUILD LABEL CSV ──────────────────────────────────────────────────

def build_csv(
    splits: dict[str, Path],
    class_mapping: dict[str, str],
    out_csv: Path,
) -> list[dict]:
    """
    Walk each split, look up the disease label for every class folder,
    and write one row per image to the output CSV.

    CSV columns:
        split         - "train" or "val"
        image_path    - relative path from project root (forward slashes)
        original_class- folder name as-is (e.g. "tomato late blight")
        disease_label - disease only     (e.g. "late blight")
    
    Keeping original_class in the CSV is intentional: it lets you trace
    back which plant host an image came from during error analysis.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    unmapped = []

    for split_name, split_path in splits.items():
        if not split_path.exists():
            print(f"  WARNING: {split_path} not found — skipping {split_name}")
            continue

        class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
        print(f"\n  [{split_name}] {len(class_dirs)} class folders found")

        for class_dir in class_dirs:
            folder_name = class_dir.name.strip().lower()

            # Look up disease label from mapping
            disease = class_mapping.get(folder_name)
            if disease is None:
                print(f"  WARNING: '{class_dir.name}' not in class_mapping.csv — skipping folder")
                unmapped.append(class_dir.name)
                continue

            # Collect all valid images in this folder
            images = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in VALID_EXTENSIONS
            ]

            for img_path in images:
                rows.append({
                    "split":          split_name,
                    "image_path":     img_path.as_posix(),
                    "original_class": class_dir.name,
                    "disease_label":  disease,
                })

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "image_path", "original_class", "disease_label"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Written {len(rows)} image rows to {out_csv}")

    if unmapped:
        print(f"\n  {len(unmapped)} folder(s) had no mapping — check class_mapping.csv:")
        for u in unmapped:
            print(f"    - {u}")

    return rows

# ─── STEP 3: SUMMARIZE ────────────────────────────────────────────────────────

def summarize(rows: list[dict]):
    """
    Print a per-disease breakdown of image counts in train vs val.

    Flags:
      ← train only  : class has no val images (can't validate on it)
      ← val only    : class has no train images (model never saw it)
    
    This is important to check before training — heavy class imbalance
    or missing val classes directly affect how you interpret your metrics.
    """
    train_counts = Counter(r["disease_label"] for r in rows if r["split"] == "train")
    val_counts   = Counter(r["disease_label"] for r in rows if r["split"] == "val")
    all_diseases = sorted(set(train_counts) | set(val_counts))

    print(f"\n{'Disease':<48} {'Train':>6} {'Val':>6}  Note")
    print("─" * 75)

    for disease in all_diseases:
        t = train_counts.get(disease, 0)
        v = val_counts.get(disease, 0)
        note = ""
        if t == 0:
            note = "← val only"
        elif v == 0:
            note = "← train only"
        print(f"{disease:<48} {t:>6} {v:>6}  {note}")

    print("─" * 75)
    print(f"{'TOTAL':<48} {sum(train_counts.values()):>6} {sum(val_counts.values()):>6}")
    print(f"\nUnique disease classes : {len(all_diseases)}")
    print(f"Total images           : {len(rows)}")

    # Per-disease image distribution — useful for spotting imbalance
    all_counts = Counter(r["disease_label"] for r in rows)
    most_common   = all_counts.most_common(3)
    least_common  = all_counts.most_common()[:-4:-1]
    print(f"\nMost common  : {most_common}")
    print(f"Least common : {least_common}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 1: Loading class mapping ===")
    class_mapping = load_class_mapping(CLASS_MAP)

    print("\n=== Step 2: Building label CSV ===")
    rows = build_csv(
        splits={"train": RAW_TRAIN, "val": RAW_VAL},
        class_mapping=class_mapping,
        out_csv=OUT_CSV,
    )

    print("\n=== Step 3: Dataset summary ===")
    summarize(rows)