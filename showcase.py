"""
showcase.py
-----------
Live detection showcase for CodeKids.
Draws bounding boxes around detected objects with big colorful labels.

Controls:
  M  → toggle mode (SORTING / CANDY HUNT)
  Q  → quit

Prerequisites:
    pip install ultralytics opencv-python numpy
    model_detection.pt must exist (run train_detection.py first)
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "model_detection.pt"
CAMERA_INDEX = 0
CONF_THRESH  = 0.50
CLASSES      = ["trash", "recycling", "candy"]
# ─────────────────────────────────────────────────────────────────────────────

CLASS_CONFIG = {
    "trash": {
        "color"  : (60,  60,  220),
        "label"  : "TRASH",
        "message": "Throw it away!",
        "bg"     : (30, 30, 100),
    },
    "recycling": {
        "color"  : (50,  200, 50),
        "label"  : "RECYCLE",
        "message": "Put it in the blue bin!",
        "bg"     : (20, 80,  20),
    },
    "candy": {
        "color"  : (220, 80,  220),
        "label"  : "CANDY!",
        "message": "You found the candy!",
        "bg"     : (100, 30, 100),
    },
}

MODES = ["SORTING", "CANDY HUNT"]
MODE_COLORS = {
    "SORTING"    : (50, 200, 50),
    "CANDY HUNT" : (220, 80, 220),
}


def draw_detection(frame, box, cls_name, confidence):
    cfg   = CLASS_CONFIG[cls_name]
    color = cfg["color"]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    pulse = int(2 * abs(np.sin(time.time() * 4)))
    thick = 3 + pulse

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    corner_len = 20
    for cx, cy, dx, dy in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + dx*corner_len, cy), color, thick+1)
        cv2.line(frame, (cx, cy), (cx, cy + dy*corner_len), color, thick+1)

    label_text = f"{cfg['label']}  {confidence*100:.0f}%"
    font       = cv2.FONT_HERSHEY_DUPLEX
    scale      = 0.8
    (tw, th), baseline = cv2.getTextSize(label_text, font, scale, 2)
    label_y = max(y1 - 10, th + 10)
    cv2.rectangle(frame,
                  (x1, label_y - th - 8),
                  (x1 + tw + 10, label_y + baseline),
                  cfg["bg"], -1)
    cv2.rectangle(frame,
                  (x1, label_y - th - 8),
                  (x1 + tw + 10, label_y + baseline),
                  color, 2)
    cv2.putText(frame, label_text,
                (x1 + 5, label_y),
                font, scale, color, 2, cv2.LINE_AA)


def draw_bottom_banner(frame, detections):
    if not detections:
        return
    h, w = frame.shape[:2]
    bar_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    best    = max(detections, key=lambda d: d[1])
    cls_name = best[0]
    cfg     = CLASS_CONFIG[cls_name]
    font    = cv2.FONT_HERSHEY_DUPLEX

    cv2.putText(frame, cfg["message"],
                (w//2 - 200, h - bar_h + 50),
                font, 1.4, cfg["color"], 3, cv2.LINE_AA)

    items = ", ".join([f"{d[0].capitalize()} ({d[1]*100:.0f}%)" for d in detections])
    cv2.putText(frame, items,
                (w//2 - min(len(items)*5, w//2 - 20), h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)


def draw_scanning(frame):
    h, w = frame.shape[:2]
    y = int((time.time() % 2) / 2 * h)
    cv2.line(frame, (0, y), (w, y), (0, 200, 255), 2)
    cv2.putText(frame, "SCANNING...",
                (w//2 - 100, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)


def draw_mode_badge(frame, mode):
    color = MODE_COLORS[mode]
    cv2.rectangle(frame, (0, 0), (270, 45), (20, 20, 20), -1)
    cv2.putText(frame, f"MODE: {mode}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def main():
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        print("Run train_detection.py first.")
        return

    print("Loading detection model...")
    model = YOLO(MODEL_PATH)
    print("Ready! Starting camera...")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_INDEX}.")
        print("Change CAMERA_INDEX at the top of showcase.py")
        return

    mode_idx = 0
    print("Live! M=Switch mode  Q=Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mode = MODES[mode_idx]

        results    = model.predict(frame, conf=CONF_THRESH,
                                   verbose=False, imgsz=640)
        boxes      = results[0].boxes
        detections = []

        for box in boxes:
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "trash"
            xyxy     = box.xyxy[0].tolist()

            if mode == "CANDY HUNT" and cls_name != "candy":
                continue

            draw_detection(frame, xyxy, cls_name, conf)
            detections.append((cls_name, conf))

        if detections:
            draw_bottom_banner(frame, detections)
        else:
            if mode == "CANDY HUNT":
                h, w = frame.shape[:2]
                cv2.putText(frame, "Find the candy!",
                            (w//2 - 160, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (220, 80, 220), 2, cv2.LINE_AA)
            else:
                draw_scanning(frame)

        draw_mode_badge(frame, mode)
        h, w = frame.shape[:2]
        cv2.putText(frame, "M=Switch Mode  Q=Quit",
                    (w - 290, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow("CodeKids Vision", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode_idx = (mode_idx + 1) % len(MODES)
            print(f"Mode: {MODES[mode_idx]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
