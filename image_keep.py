import cv2
from pathlib import Path

IMG_DIR = Path("labeled/images")
LBL_DIR = Path("labeled/labels")
CLASSES = ["trash", "recycling", "candy"]
COLORS  = [(60, 60, 220), (50, 200, 50), (220, 80, 220)]

for img_path in sorted(IMG_DIR.glob("*.jpg")):
    frame = cv2.imread(str(img_path))
    h, w = frame.shape[:2]

    lbl_path = LBL_DIR / (img_path.stem + ".txt")
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            cls, cx, cy, bw, bh = map(float, line.split())
            cls = int(cls)
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[cls], 2)
            cv2.putText(frame, CLASSES[cls], (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[cls], 2)

    cv2.putText(frame, img_path.name, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, "D=Delete  SPACE=Keep  Q=Quit", (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.imshow("Review", frame)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        img_path.unlink()
        lbl_path.unlink()
        print(f"Deleted {img_path.name}")

cv2.destroyAllWindows()