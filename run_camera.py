import cv2
import numpy as np
import json

with open("camera_params.json") as f:
    params = json.load(f)

K = np.array(params["camera_matrix"])
dist = np.array(params["dist_coeff"])

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Camera not found")

ret, frame = cap.read()
h, w = frame.shape[:2]

newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

mapx, mapy = cv2.initUndistortRectifyMap(
    K,
    dist,
    None,
    newK,
    (w,h),
    cv2.CV_32FC1
)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    undistorted = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

    x,y,w_roi,h_roi = roi
    undistorted = undistorted[y:y+h_roi, x:x+w_roi]

    cv2.imshow("Raw",frame)
    cv2.imshow("Undistorted",undistorted)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
