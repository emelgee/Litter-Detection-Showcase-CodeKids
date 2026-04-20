"""
setup.py
--------
Automatically downloads and prepares training data from online datasets.
Handles trash/recycling from TrashNet + Kaggle Garbage Classification,
and prompts you to collect candy images manually (since specific candy
types aren't in public datasets).

Run this ONCE before train.py.

Prerequisites:
    pip install ultralytics opencv-python numpy requests kaggle

For Kaggle downloads you need a Kaggle API key:
  1. Go to kaggle.com → Account → Create New API Token
  2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json
  3. On Mac/Linux: chmod 600 ~/.kaggle/kaggle.json
"""

import os
import sys
import shutil
import zipfile
import random
import subprocess
from pathlib import Path

DATA_DIR = Path("data")
CLASSES  = ["trash", "recycling", "candy"]

# How many images to pull per class from online datasets (more = better accuracy)
IMAGES_PER_CLASS = 300

# Mapping from dataset folder names → our class names
KAGGLE_CLASS_MAP = {
    "trash"     : "trash",
    "plastic"   : "recycling",
    "metal"     : "recycling",
    "glass"     : "recycling",
    "cardboard" : "recycling",
    "paper"     : "recycling",
}

TRASHNET_CLASS_MAP = {
    "trash"     : "trash",
    "plastic"   : "recycling",
    "metal"     : "recycling",
    "glass"     : "recycling",
    "cardboard" : "recycling",
    "paper"     : "recycling",
}


def banner(msg):
    print("\n" + "=" * 55)
    print(f"  {msg}")
    print("=" * 55)


def ensure_dirs():
    for cls in CLASSES:
        (DATA_DIR / cls).mkdir(parents=True, exist_ok=True)


def count_images(cls):
    return len(list((DATA_DIR / cls).glob("*.jpg"))) + \
           len(list((DATA_DIR / cls).glob("*.png")))


def copy_images(src_dir, dest_class, limit=None):
    """Copy images from src_dir into data/<dest_class>/"""
    src = Path(src_dir)
    if not src.exists():
        return 0
    imgs = list(src.glob("*.jpg")) + list(src.glob("*.png")) + \
           list(src.glob("*.jpeg"))
    random.shuffle(imgs)
    if limit:
        imgs = imgs[:limit]
    dest = DATA_DIR / dest_class
    copied = 0
    for img in imgs:
        shutil.copy(img, dest / img.name)
        copied += 1
    return copied


def download_kaggle_dataset():
    """Download garythung/trashnet via Kaggle API."""
    banner("Downloading Kaggle Garbage Classification Dataset")
    try:
        import kaggle  # noqa — just checking it's installed
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("Run: pip install kaggle")
        print("Then set up your API key (see file header).")
        return False

    dl_dir = Path("downloads/kaggle_garbage")
    dl_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "asdasdasasdas/garbage-classification",
            "-p", str(dl_dir), "--unzip"
        ], check=True)
    except subprocess.CalledProcessError:
        # Try alternate popular dataset
        try:
            subprocess.run([
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", "mostafaabla/garbage-classification",
                "-p", str(dl_dir), "--unzip"
            ], check=True)
        except subprocess.CalledProcessError:
            print("ERROR: Could not download Kaggle dataset.")
            print("Check your kaggle.json API key is set up correctly.")
            return False

    # Walk all subdirs and copy into our class structure
    total = {cls: 0 for cls in CLASSES}
    per_source_limit = IMAGES_PER_CLASS // 3  # spread across categories

    for folder in dl_dir.rglob("*"):
        if not folder.is_dir():
            continue
        folder_name = folder.name.lower()
        if folder_name in KAGGLE_CLASS_MAP:
            dest_cls = KAGGLE_CLASS_MAP[folder_name]
            n = copy_images(folder, dest_cls, limit=per_source_limit)
            total[dest_cls] += n
            if n > 0:
                print(f"  {folder_name} → {dest_cls}: {n} images")

    print("\nKaggle totals so far:")
    for cls, n in total.items():
        print(f"  {cls}: {n}")
    return True


def download_trashnet():
    """Download TrashNet from GitHub releases."""
    banner("Downloading TrashNet Dataset")
    import urllib.request

    url     = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    dl_dir  = Path("downloads/trashnet")
    dl_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dl_dir / "trashnet.zip"

    if not zip_path.exists():
        print(f"Downloading from {url} ...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"ERROR downloading TrashNet: {e}")
            print("You may need to download it manually from:")
            print("  https://github.com/garythung/trashnet")
            return False

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dl_dir)

    # TrashNet extracts to dataset-resized/<class>/
    extracted = dl_dir / "dataset-resized"
    if not extracted.exists():
        # Try one level deeper
        for sub in dl_dir.rglob("dataset-resized"):
            extracted = sub
            break

    per_source_limit = IMAGES_PER_CLASS // 2

    for folder in extracted.iterdir():
        if not folder.is_dir():
            continue
        folder_name = folder.name.lower()
        if folder_name in TRASHNET_CLASS_MAP:
            dest_cls = TRASHNET_CLASS_MAP[folder_name]
            n = copy_images(folder, dest_cls, limit=per_source_limit)
            if n > 0:
                print(f"  {folder_name} → {dest_cls}: {n} images")

    return True


def candy_instructions():
    """Guide user to collect candy images manually."""
    banner("Candy Images — Manual Collection Required")
    current = count_images("candy")
    print(f"Current candy images: {current}")
    print()
    print("Public datasets don't have specific candy types like")
    print("Starbursts, Dum Dums, or Hershey's Kisses, so you'll")
    print("need to collect these yourself using collect_data.py.")
    print()
    print("Steps:")
    print("  1. Run:  python collect_data.py")
    print("  2. Press C to save candy frames")
    print("  3. Aim for 80+ images — vary angles, backgrounds, lighting")
    print()
    if current >= 80:
        print(f"✓ You already have {current} candy images — you're good to go!")
    else:
        needed = 80 - current
        print(f"⚠  You need ~{needed} more candy images before training.")


def print_summary():
    banner("Dataset Summary")
    ready = True
    for cls in CLASSES:
        n = count_images(cls)
        status = "✓" if n >= 50 else "⚠ "
        if n < 50:
            ready = False
        print(f"  {status} {cls}: {n} images {'(need 50+)' if n < 50 else ''}")

    print()
    if ready:
        print("✓ All classes have enough images.")
        print("  Run:  python train.py")
    else:
        print("⚠  Some classes need more images before training.")
        print("  Collect candy images with:  python collect_data.py")


def main():
    banner("CodeKids Vision — Auto Dataset Setup")
    ensure_dirs()

    # Try TrashNet first (no API key needed)
    trashnet_ok = download_trashnet()

    # Then try Kaggle for more variety
    kaggle_ok = False
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        kaggle_ok = download_kaggle_dataset()
    else:
        print("\nKaggle API key not found — skipping Kaggle dataset.")
        print("(Optional) To add more data:")
        print("  1. Go to kaggle.com → Account → Create New API Token")
        print("  2. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("  3. Re-run this script")

    if not trashnet_ok and not kaggle_ok:
        print("\nERROR: Could not download any datasets automatically.")
        print("Please collect all images manually using collect_data.py")
        return

    candy_instructions()
    print_summary()


if __name__ == "__main__":
    main()
