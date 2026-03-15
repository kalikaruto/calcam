import cv2
import numpy as np
import glob
import json
import config

rows = config.ROWS
cols = config.COLS
square_size = config.SQUARE_SIZE

objp = np.zeros((rows*cols,3), np.float32)
objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob(f"{config.CALIB_IMAGE_DIR}/{config.IMAGE_EXTENSION}")

if len(images) == 0:
    raise RuntimeError("No calibration images found")

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    config.CRITERIA_MAX_ITER,
    config.CRITERIA_EPS
)

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,(cols,rows),None)

    if ret:

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            config.CORNER_WINDOW,
            config.ZERO_ZONE,
            criteria
        )

        imgpoints.append(corners2)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

data = {
    "camera_matrix": K.tolist(),
    "dist_coeff": dist.tolist()
}

with open(config.CALIB_FILE,"w") as f:
    json.dump(data,f,indent=4)
