"""
train.py
--------
Fine-tunes a YOLOv8 image classifier on your collected data.

Prerequisites:
    pip install ultralytics

Data layout expected (created by collect_data.py):
    data/
        trash/        *.jpg
        recycling/    *.jpg
        candy/        *.jpg

This uses YOLOv8's classify mode which:
  - Starts from ImageNet pretrained weights
  - Fine-tunes only the head for fast training
  - Exports a model.pt you can load in showcase.py

Typical training time: 5-15 min on CPU for 100 imgs/class at 30 epochs.
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
DATASET_DIR   = Path("dataset")          # train/val split goes here
CLASSES       = ["trash", "recycling", "candy"]
VAL_SPLIT     = 0.2                      # 20% validation
EPOCHS        = 50
IMAGE_SIZE    = 224
MODEL_BASE    = "yolov8n-cls.pt"         # nano = fastest; swap to yolov8s-cls.pt for better accuracy
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset():
    """Split raw data into train/val folders for Ultralytics."""
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val"]:
        for cls in CLASSES:
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    total = 0
    for cls in CLASSES:
        images = list((DATA_DIR / cls).glob("*.jpg"))
        random.shuffle(images)
        if len(images) == 0:
            print(f"WARNING: No images found for class '{cls}'")
            continue

        n_val = max(1, int(len(images) * VAL_SPLIT))
        val_imgs   = images[:n_val]
        train_imgs = images[n_val:]

        for img in train_imgs:
            shutil.copy(img, DATASET_DIR / "train" / cls / img.name)
        for img in val_imgs:
            shutil.copy(img, DATASET_DIR / "val" / cls / img.name)

        print(f"  {cls}: {len(train_imgs)} train / {len(val_imgs)} val")
        total += len(images)

    print(f"\nTotal images: {total}")
    return total

def train():
    print("=" * 50)
    print("  CodeKids Vision Trainer")
    print("=" * 50)

    print("\nBuilding train/val split...")
    total = build_dataset()

    if total == 0:
        print("\nERROR: No images found. Run collect_data.py first.")
        return

    print(f"\nLoading base model: {MODEL_BASE}")
    model = YOLO(MODEL_BASE)

    print(f"\nTraining for {EPOCHS} epochs...")
    results = model.train(
        data=str(DATASET_DIR),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=16,
        patience=10,            # early stopping
        augment=True,           # random flips, color jitter, etc.
        plots=True,
        project="runs",
        name="codekids",
        exist_ok=True,
    )

    best_model = Path("runs/codekids/weights/best.pt")
    if best_model.exists():
        shutil.copy(best_model, "model.pt")
        print(f"\n✓ Model saved to model.pt")
        print(f"  Top-1 accuracy: {results.results_dict.get('metrics/accuracy_top1', 'N/A')}")
    else:
        print("\nERROR: Training failed, no weights found.")

    print("\nDone! Run showcase.py to start the live demo.")

if __name__ == "__main__":
    train()
