import cv2
import os

SAVE_DIR = "calib_images"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    raise RuntimeError("Camera not found")

img_id = 0

print("Press c to capture image")
print("Press q to quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):   # ESC
        break

    if key == ord('c'):   # SPACE
        filename = f"{SAVE_DIR}/img_{img_id}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        img_id += 1

cap.release()
cv2.destroyAllWindows()
