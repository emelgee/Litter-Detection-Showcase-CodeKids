"""
collect_data.py
---------------
Webcam-based image collector for training data.

Controls:
  T  → save frame to data/trash/
  R  → save frame to data/recycling/
  C  → save frame to data/candy/
  Q  → quit

Aim to collect 80-150 images per class for best results.
Mix backgrounds, lighting, angles, and distances.
"""

import cv2
import os
import time

CLASSES = ["trash", "recycling", "candy"]
KEY_MAP = {ord('t'): "trash", ord('r'): "recycling", ord('c'): "candy"}
DATA_DIR = "data"

for cls in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

counts = {cls: len(os.listdir(os.path.join(DATA_DIR, cls))) for cls in CLASSES}

cap = cv2.VideoCapture(1)  # Change to 0 if USB cam isn't index 1
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Data Collector Ready")
print("T=Trash | R=Recycling | C=Candy | Q=Quit")
for cls in CLASSES:
    print(f"  {cls}: {counts[cls]} images")

flash = None
flash_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # HUD overlay
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (400, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    cv2.putText(display, "T=Trash  R=Recycle  C=Candy  Q=Quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    for i, cls in enumerate(CLASSES):
        colors = {"trash": (80, 80, 255), "recycling": (80, 255, 80), "candy": (255, 80, 200)}
        cv2.putText(display, f"{cls.capitalize()}: {counts[cls]} imgs",
                    (10, 55 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[cls], 2)

    # Flash feedback
    if flash and (time.time() - flash_time) < 0.4:
        colors = {"trash": (80, 80, 255), "recycling": (80, 255, 80), "candy": (255, 80, 200)}
        cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]),
                      colors[flash], 8)
        cv2.putText(display, f"SAVED: {flash.upper()}",
                    (display.shape[1]//2 - 120, display.shape[0]//2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, colors[flash], 3)
    else:
        flash = None

    cv2.imshow("Data Collector", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in KEY_MAP:
        cls = KEY_MAP[key]
        timestamp = int(time.time() * 1000)
        filename = os.path.join(DATA_DIR, cls, f"{cls}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        counts[cls] += 1
        flash = cls
        flash_time = time.time()
        print(f"Saved {filename} ({counts[cls]} total)")

cap.release()
cv2.destroyAllWindows()
print("\nFinal counts:")
for cls in CLASSES:
    print(f"  {cls}: {counts[cls]} images")
