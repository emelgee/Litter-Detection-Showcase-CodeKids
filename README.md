# CodeKids Vision — Setup & Usage

## Install dependencies
```bash
pip install ultralytics opencv-python numpy requests kaggle
```

---

## Step 1 — Auto-Download Datasets
```bash
python setup.py
```
This automatically downloads trash/recycling images from:
- **TrashNet** (no account needed)
- **Kaggle Garbage Classification** (optional, needs API key — see below)

**Kaggle API key setup (optional but recommended for more data):**
1. Go to kaggle.com → Account → Create New API Token
2. Move the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
3. On Mac/Linux run: `chmod 600 ~/.kaggle/kaggle.json`
4. Re-run `python setup.py`

---

## Step 2 — Collect Candy Images (Manual, ~15 min)
```bash
python collect_data.py
```
Public datasets don't have Starbursts, Dum Dums, or Hershey's Kisses,
so you need to photograph these yourself.

- Press **C** to save candy frames
- Aim for **80+ images** — vary angles, backgrounds, and distances
- Press **Q** when done

---

## Step 3 — Train the Model
```bash
python train.py
```
- Takes ~5–15 minutes on CPU
- Outputs `model.pt` when done
- Aim for **85%+ accuracy** before the showcase

---

## Step 4 — Run the Showcase
```bash
python showcase.py
```
- **M** → switch between SORTING mode and CANDY HUNT mode
- **Q** → quit

**If the camera doesn't open**, change `CAMERA_INDEX = 1` to `CAMERA_INDEX = 0`
at the top of `showcase.py`.

---

## Adding More Candy Types
To add Skittles, M&Ms, etc.:
1. Add a new folder: `data/skittles/`
2. Collect images with `collect_data.py` (add a new key binding)
3. Add `"skittles"` to the `CLASSES` list in all files
4. Add an entry to `CLASS_CONFIG` in `showcase.py`
5. Re-run `train.py`

---

## File Overview
| File | Purpose |
|------|---------|
| `setup.py` | Auto-downloads trash/recycling datasets |
| `collect_data.py` | Webcam tool to capture candy training images |
| `train.py` | Fine-tunes YOLOv8 on your images |
| `showcase.py` | Live demo with big labels + candy hunt mode |
