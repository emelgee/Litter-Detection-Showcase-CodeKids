# CodeKids Vision — Setup & Usage

## Install dependencies
```bash
pip install ultralytics opencv-python numpy
```

---

## Step 1 — Collect Training Data
```bash
python collect_data.py
```
Point your webcam at objects and press:
- **T** → save as Trash
- **R** → save as Recycling
- **C** → save as Candy (e.g. Starbursts)
- **Q** → quit

**Tips for best results:**
- Aim for 80–150 images per class
- Vary lighting, backgrounds, angles, and distances
- Hold the object in different positions
- Include some "almost candy" objects (wrappers, foil) in trash/recycling to help it distinguish

---

## Step 2 — Train the Model
```bash
python train.py
```
- Takes ~5–15 minutes on CPU
- Outputs `model.pt` when done
- Check the printed accuracy — aim for 85%+ before the showcase

**If accuracy is low:**
- Collect more images (especially for the weak class)
- Re-run train.py (it rebuilds the split each time)

---

## Step 3 — Run the Showcase
```bash
python showcase.py
```
- **M** → switch between SORTING mode and CANDY HUNT mode
- **Q** → quit

**If the camera doesn't open**, change `CAMERA_INDEX = 1` to `CAMERA_INDEX = 0` at the top of showcase.py.

---

## Adding More Candy Types
To add Skittles, M&Ms, etc.:
1. Add a new folder: `data/skittles/`
2. Collect images with collect_data.py (you'll need to add a key binding for it)
3. Add `"skittles"` to the `CLASSES` list in all three files
4. Add an entry to `CLASS_CONFIG` in showcase.py
5. Re-run train.py

---

## File Overview
| File | Purpose |
|------|---------|
| `collect_data.py` | Webcam tool to capture training images |
| `train.py` | Fine-tunes YOLOv8 on your images |
| `showcase.py` | Live demo with big labels + candy hunt mode |
