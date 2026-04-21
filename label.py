"""
label.py
--------
Webcam-based bounding box labeling tool.
Capture a frame, draw a box around the object, assign a class.

Controls:
  SPACE       → capture frame from webcam
  Left-click  → start drawing box
  Left-drag   → draw box
  T           → label current box as Trash
  R           → label current box as Recycling
  C           → label current box as Candy
  U           → undo last box on current frame
  N           → next frame (save current and go back to webcam)
  Q           → quit

Saved structure:
  labeled/
    images/   *.jpg
    labels/   *.txt  (YOLO format: class cx cy w h, normalized)
"""

import cv2
import os
import time
import json
from pathlib import Path

CLASSES     = ["trash", "recycling", "candy"]
KEY_CLASS   = {ord('t'): 0, ord('r'): 1, ord('c'): 2}
COLORS      = [(60, 60, 220), (50, 200, 50), (220, 80, 220)]
IMG_DIR     = Path("labeled/images")
LBL_DIR     = Path("labeled/labels")
CAMERA_INDEX = 1  # Change to 1 if wrong camera

IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

# ── State ──────────────────────────────────────────────────────────────────
drawing     = False
start_pt    = None
current_box = None   # (x1,y1,x2,y2) being drawn
boxes       = []     # list of (x1,y1,x2,y2,class_id) confirmed on this frame
frame_live  = None   # live webcam frame
frame_edit  = None   # captured frame being labeled
mode        = "live" # "live" or "edit"

def normalize_box(x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h
    return cx, cy, bw, bh

def draw_boxes(img, boxes, current_box=None):
    out = img.copy()
    for (x1, y1, x2, y2, cls_id) in boxes:
        color = COLORS[cls_id]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, CLASSES[cls_id], (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if current_box:
        x1, y1, x2, y2 = current_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 0), 2)
    return out

def draw_hud(img, mode, boxes):
    out = img.copy()
    h, w = out.shape[:2]
    # Bottom bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h-45), (w, h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    if mode == "live":
        msg = "SPACE=Capture frame    Q=Quit"
    else:
        msg = "T=Trash  R=Recycle  C=Candy  U=Undo  N=Next frame (no boxes=background)  Q=Quit"
    cv2.putText(out, msg, (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    # Top bar
    overlay2 = out.copy()
    cv2.rectangle(overlay2, (0,0), (w, 40), (20,20,20), -1)
    cv2.addWeighted(overlay2, 0.7, out, 0.3, 0, out)

    total = len(list(IMG_DIR.glob("*.jpg")))
    status = f"MODE: {mode.upper()}   |   Boxes this frame: {len(boxes)}   |   Total saved: {total}"
    cv2.putText(out, status, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    return out

def mouse_cb(event, x, y, flags, param):
    global drawing, start_pt, current_box
    if mode != "edit":
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing  = True
        start_pt = (x, y)
        current_box = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_box = (start_pt[0], start_pt[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing     = False
        current_box = (start_pt[0], start_pt[1], x, y)

def save_frame(frame, boxes):
    ts   = int(time.time() * 1000)
    name = f"frame_{ts}"
    cv2.imwrite(str(IMG_DIR / f"{name}.jpg"), frame)

    if not boxes:
        # No label file = background/negative sample
        print(f"Saved {name} as background (no boxes).")
        return

    h, w = frame.shape[:2]
    lines = []
    for (x1, y1, x2, y2, cls_id) in boxes:
        cx, cy, bw, bh = normalize_box(x1, y1, x2, y2, w, h)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(LBL_DIR / f"{name}.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {name} with {len(boxes)} box(es).")

def main():
    global frame_live, frame_edit, mode, boxes, current_box, drawing

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_INDEX}.")
        print("Change CAMERA_INDEX at the top of label.py")
        return

    cv2.namedWindow("Labeler")
    cv2.setMouseCallback("Labeler", mouse_cb)

    print("Labeling Tool Ready")
    print("SPACE=Capture  T=Trash  R=Recycle  C=Candy  U=Undo  N=Next  Q=Quit")
    print("Tip: Press N with no boxes to save a background (negative) sample.")

    while True:
        if mode == "live":
            ret, frame_live = cap.read()
            if not ret:
                break
            display = draw_hud(frame_live, mode, [])

        else:  # edit mode
            display = draw_boxes(frame_edit, boxes, current_box)
            display = draw_hud(display, mode, boxes)

        cv2.imshow("Labeler", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' ') and mode == "live":
            frame_edit = frame_live.copy()
            boxes      = []
            current_box = None
            mode       = "edit"
            print("Frame captured. Draw boxes and label, or press N to save as background.")

        elif key in KEY_CLASS and mode == "edit" and current_box:
            x1,y1,x2,y2 = current_box
            if abs(x2-x1) > 10 and abs(y2-y1) > 10:
                cls_id = KEY_CLASS[key]
                boxes.append((min(x1,x2), min(y1,y2),
                               max(x1,x2), max(y1,y2), cls_id))
                current_box = None
                print(f"  Box added: {CLASSES[cls_id]} ({len(boxes)} total)")
            else:
                print("  Box too small — try again.")

        elif key == ord('u') and mode == "edit":
            if boxes:
                removed = boxes.pop()
                print(f"  Undid last box ({CLASSES[removed[4]]})")

        elif key == ord('n') and mode == "edit":
            save_frame(frame_edit, boxes)
            boxes       = []
            current_box = None
            mode        = "live"

    cap.release()
    cv2.destroyAllWindows()

    total = len(list(IMG_DIR.glob("*.jpg")))
    print(f"\nDone. {total} labeled frames saved to labeled/")

if __name__ == "__main__":
    main()