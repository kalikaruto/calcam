import cv2
import os
import config

os.makedirs(config.CALIB_IMAGE_DIR, exist_ok=True)

cap = cv2.VideoCapture(config.CAMERA_ID, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.PIXEL_FORMAT))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

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

    if key == ord('q'):
        break

    if key == ord('c'):
        filename = f"{config.CALIB_IMAGE_DIR}/img_{img_id}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        img_id += 1

cap.release()
cv2.destroyAllWindows()
