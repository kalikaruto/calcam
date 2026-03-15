import cv2
import numpy as np
import json
import config

with open(config.CALIB_FILE) as f:
    params = json.load(f)

K = np.array(params["camera_matrix"])
dist = np.array(params["dist_coeff"])

cap = cv2.VideoCapture(config.CAMERA_ID, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.PIXEL_FORMAT))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
