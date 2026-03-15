import cv2
import numpy as np
import glob
import json

rows = 10
cols = 9
square_size = 20.0

objp = np.zeros((rows*cols,3), np.float32)
objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob("calib_images/*.jpg")

if len(images) == 0:
    raise RuntimeError("No calibration images found")

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,(cols,rows),None)

    if ret:

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11,11),
            (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
        )

        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img,(cols,rows),corners2,ret)
        cv2.imshow("Corners",img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

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

with open("camera_params.json","w") as f:
    json.dump(data,f,indent=4)

print("Calibration complete")
print("Camera Matrix:\n",K)
print("Distortion:\n",dist)
