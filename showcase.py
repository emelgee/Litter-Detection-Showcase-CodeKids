"""
showcase.py
-----------
Live demo for the CodeKids showcase.
Shows webcam feed with big colorful classification labels.

Controls:
  M  → toggle mode (SORTING / CANDY HUNT)
  Q  → quit

Prerequisites:
    pip install ultralytics opencv-python numpy

Make sure model.pt exists (run train.py first).
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "model.pt"
CAMERA_INDEX = 1          # change to 0 if needed
CONF_THRESH  = 0.55       # minimum confidence to show a label
CLASSES      = ["trash", "recycling", "candy"]
# ─────────────────────────────────────────────────────────────────────────────

CLASS_CONFIG = {
    "trash": {
        "color"  : (60,  60,  220),   # red-ish (BGR)
        "emoji"  : "🗑️",
        "label"  : "TRASH",
        "message": "Throw it away!",
        "bg"     : (30, 30, 100),
    },
    "recycling": {
        "color"  : (50,  200, 50),    # green
        "emoji"  : "♻️",
        "label"  : "RECYCLE",
        "message": "Put it in the blue bin!",
        "bg"     : (20, 80,  20),
    },
    "candy": {
        "color"  : (220, 80,  220),   # pink/purple
        "emoji"  : "🍬",
        "label"  : "CANDY FOUND!",
        "message": "You found the candy!",
        "bg"     : (100, 30, 100),
    },
}

MODES = ["SORTING", "CANDY HUNT"]
MODE_COLORS = {
    "SORTING"    : (50, 200, 50),
    "CANDY HUNT" : (220, 80, 220),
}


def draw_big_label(frame, cls_name, confidence):
    """Draws the large bottom banner with class info."""
    cfg   = CLASS_CONFIG[cls_name]
    h, w  = frame.shape[:2]
    bar_h = 160

    # Semi-transparent background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), cfg["bg"], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Colored border at top of bar
    cv2.rectangle(frame, (0, h - bar_h), (w, h - bar_h + 6), cfg["color"], -1)

    # Main label
    label = cfg["label"]
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 2.8
    thick = 4
    (lw, lh), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.putText(frame, label,
                (w // 2 - lw // 2, h - bar_h + 70),
                font, scale, cfg["color"], thick, cv2.LINE_AA)

    # Sub-message
    msg   = cfg["message"]
    scale2 = 0.9
    (mw, _), _ = cv2.getTextSize(msg, font, scale2, 2)
    cv2.putText(frame, msg,
                (w // 2 - mw // 2, h - bar_h + 115),
                font, scale2, (230, 230, 230), 2, cv2.LINE_AA)

    # Confidence pill
    conf_text = f"{confidence*100:.0f}% sure"
    cv2.putText(frame, conf_text,
                (w // 2 - 60, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)


def draw_detection_box(frame, cls_name, confidence):
    """Draws an animated border around the whole frame."""
    cfg   = CLASS_CONFIG[cls_name]
    h, w  = frame.shape[:2]
    thick = 8
    pulse = int(4 * abs(np.sin(time.time() * 4)))  # pulsing thickness
    cv2.rectangle(frame, (0, 0), (w, h), cfg["color"], thick + pulse)


def draw_mode_badge(frame, mode):
    """Top-left mode indicator."""
    color = MODE_COLORS[mode]
    cv2.rectangle(frame, (0, 0), (260, 45), (20, 20, 20), -1)
    cv2.putText(frame, f"MODE: {mode}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def draw_scanning(frame):
    """Animated scanning line when confidence is low."""
    h, w = frame.shape[:2]
    y = int((time.time() % 2) / 2 * h)
    cv2.line(frame, (0, y), (w, y), (0, 200, 255), 2)
    cv2.putText(frame, "SCANNING...",
                (w // 2 - 100, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)


def draw_candy_hunt_overlay(frame, found):
    """Special overlay for candy hunt mode."""
    h, w = frame.shape[:2]
    if found:
        # Big celebration overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (180, 50, 180), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.putText(frame, "🍬 FOUND IT! 🍬",
                    (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 100, 255), 4, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Find the candy!",
                    (w // 2 - 160, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (220, 80, 220), 2, cv2.LINE_AA)


def main():
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: {MODEL_PATH} not found. Run train.py first.")
        return

    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded. Starting camera...")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")
        print("Try changing CAMERA_INDEX to 0 at the top of this file.")
        return

    mode_idx = 0
    print(f"\nLive! Press M to switch modes, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mode = MODES[mode_idx]

        # ── Run inference ──────────────────────────────────────────────────
        results   = model.predict(frame, verbose=False, imgsz=224)
        probs     = results[0].probs
        top_idx   = int(probs.top1)
        top_conf  = float(probs.top1conf)
        cls_name  = CLASSES[top_idx] if top_idx < len(CLASSES) else "trash"

        detected  = top_conf >= CONF_THRESH

        # ── Candy hunt mode: only react to candy ──────────────────────────
        if mode == "CANDY HUNT":
            candy_conf = float(probs.data[CLASSES.index("candy")])
            candy_found = candy_conf >= CONF_THRESH
            draw_candy_hunt_overlay(frame, candy_found)
            if candy_found:
                draw_detection_box(frame, "candy", candy_conf)
                draw_big_label(frame, "candy", candy_conf)
            else:
                draw_scanning(frame)

        # ── Sorting mode: trash or recycling (candy shown too if detected) ─
        else:
            if detected:
                draw_detection_box(frame, cls_name, top_conf)
                draw_big_label(frame, cls_name, top_conf)
            else:
                draw_scanning(frame)

        # ── Always draw mode badge & controls hint ─────────────────────────
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
