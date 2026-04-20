"""
train_detection.py
------------------
Trains a YOLOv8 object detection model on your labeled data.
Run this after label.py.

Prerequisites:
    pip install ultralytics

Expected structure (created by label.py):
    labeled/
        images/   *.jpg
        labels/   *.txt

Outputs model_detection.pt when done.
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
LABELED_DIR  = Path("labeled")
DATASET_DIR  = Path("dataset_detection")
CLASSES      = ["trash", "recycling"]   # add "candy" when ready
VAL_SPLIT    = 0.2
EPOCHS       = 100
IMAGE_SIZE   = 640                      # detection works better at 640
MODEL_BASE   = "yolov8n.pt"            # nano detection model
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset():
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    images = list((LABELED_DIR / "images").glob("*.jpg"))
    random.shuffle(images)

    if len(images) == 0:
        print("ERROR: No labeled images found. Run label.py first.")
        return 0

    n_val      = max(1, int(len(images) * VAL_SPLIT))
    val_imgs   = images[:n_val]
    train_imgs = images[n_val:]

    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        for img in imgs:
            lbl = LABELED_DIR / "labels" / (img.stem + ".txt")
            shutil.copy(img, DATASET_DIR / split / "images" / img.name)
            if lbl.exists():
                shutil.copy(lbl, DATASET_DIR / split / "labels" / lbl.name)

    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val:   {len(val_imgs)} images")
    return len(images)

def write_yaml():
    data = {
        "path"  : str(DATASET_DIR.resolve()),
        "train" : "train/images",
        "val"   : "val/images",
        "nc"    : len(CLASSES),
        "names" : CLASSES,
    }
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path

def train():
    print("=" * 55)
    print("  CodeKids Detection Trainer")
    print("=" * 55)

    print("\nBuilding dataset split...")
    total = build_dataset()
    if total == 0:
        return

    print("Writing data.yaml...")
    yaml_path = write_yaml()

    print(f"\nLoading base model: {MODEL_BASE}")
    model = YOLO(MODEL_BASE)

    print(f"Training for up to {EPOCHS} epochs...")
    results = model.train(
        data    = str(yaml_path),
        epochs  = EPOCHS,
        imgsz   = IMAGE_SIZE,
        batch   = 8,
        patience= 20,
        augment = True,
        plots   = True,
        project = "runs_detection",
        name    = "codekids",
        exist_ok= True,
    )

    # YOLOv8 detection saves to runs_detection/codekids/weights/
    best = Path("runs/detect/runs_detection/codekids/weights/best.pt")
    if best.exists():
        shutil.copy(best, "model_detection.pt")
        print(f"\n✓ Model saved to model_detection.pt")
        box_map = results.results_dict.get("metrics/mAP50(B)", "N/A")
        print(f"  mAP50: {box_map}")
        print("\n  Good results:  mAP50 > 0.7")
        print("  Great results: mAP50 > 0.85")
    else:
        print("\nERROR: Weights not found.")
        print("Check runs_detection/codekids/weights/ manually.")

    print("\nDone! Run showcase.py to start the live demo.")

if __name__ == "__main__":
    train()
