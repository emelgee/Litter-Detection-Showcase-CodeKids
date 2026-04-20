# CodeKids Vision — Setup & Usage

## Install dependencies
```bash
pip install ultralytics opencv-python numpy requests kaggle pyyaml
```

---

## Step 1 — Label Your Training Data
```bash
python label.py
```
This opens your webcam. For each item:
1. Press **SPACE** to freeze the frame
2. **Click and drag** to draw a box around the object
3. Press **T** (trash), **R** (recycling), or **C** (candy) to label it
4. Press **U** to undo a box, **N** to save and go to the next frame
5. Press **Q** when done

**Tips:**
- Aim for **60-100 labeled frames** per class
- Vary backgrounds, angles, lighting, and distances
- Label your actual demo items (can, bottle, toilet paper roll, chip bag, straw, cup)

---

## Step 2 — Train the Detection Model
```bash
python train_detection.py
```
- Takes ~15-30 min on CPU
- Outputs `model_detection.pt`
- Aim for **mAP50 > 0.70** before the showcase
- If accuracy is low, label more images and retrain

---

## Step 3 — Run the Showcase
```bash
python showcase.py
```
- Draws **bounding boxes** around detected objects in real time
- Shows the class label and confidence on each box
- **M** to switch between SORTING and CANDY HUNT mode
- **Q** to quit

**If the camera doesn't open**, change CAMERA_INDEX at the top of showcase.py and label.py.

---

## Adding Candy Later
1. Label candy images using label.py (press C to label)
2. Add "candy" back to CLASSES in train_detection.py and showcase.py
3. Re-run train_detection.py

---

## File Overview
| File | Purpose |
|------|---------|
| `label.py` | Webcam bounding box labeling tool |
| `train_detection.py` | Trains YOLOv8 detection model |
| `showcase.py` | Live demo with detection boxes + candy hunt mode |
