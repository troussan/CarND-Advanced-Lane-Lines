import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def calibrate(path, nx, ny):
    """
    Function calkulates the calibration matrix and distortion coefficients.
    Parameters:
    - path - the path to the directory containing calibration images
    - nx, ny - number of cheshboard corners

    Returns:
    - mtx - camera matrix
    - dist - distortion coefficients
    """
    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for img_name in os.listdir(path):
        img = cv2.imread(path + img_name)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add to points array
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
    # calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
    return mtx, dist

def test_calibration():
    mtx, dist = calibrate('camera_cal/', 9,  6)
    img = cv2.imread('camera_cal/calibration1.jpg')
    undistorted = cv2.undistort(img, mtx, dist, None, mtx) 

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.show()
    
if __name__ == "__main__":
    test_calibration()

