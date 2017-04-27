import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import threshold as thr
import calibration as cal

SOURCE_POINTS=np.float32([[255, 685],  [1055, 685],  [660, 435], [621, 435]])
DESTINATION_POINTS=np.float32([[255, 720],  [1055, 720],  [1055, 0], [255, 0]])

def perspective(src=SOURCE_POINTS, dst=DESTINATION_POINTS):
    M = cv2.getPerspectiveTransform(src, dst)
    R = cv2.getPerspectiveTransform(dst, src)
    return M, R

def transform(img, matrix):
    return cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

def test_perspective_color(img_path):
    mtx, dist =cal.calibrate('camera_cal/', 9,  6)
    img = plt.imread(img_path)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx) 
    M, _ = perspective()
    transformed = transform(undistorted, M)

    cv2.polylines(undistorted, [SOURCE_POINTS.reshape((-1,1,2)).astype(np.int32)], True, (255,0,0),3)
    transf_pts = DESTINATION_POINTS.reshape((-1,1,2)).astype(np.int32)
    cv2.polylines(transformed, [transf_pts], True, (0,0,255),3)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(undistorted)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(transformed)
    ax2.set_title(' Transformed result', fontsize=50)
    plt.show()

def test_perspective_threshold(img_path):
    mtx, dist =cal.calibrate('camera_cal/', 9,  6)
    img = plt.imread(img_path)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx) 
    t = thr.threshold(undistorted)
    M, _ = perspective()
    transformed = transform(t, M)*255

    transf_pts = DESTINATION_POINTS.reshape((-1,1,2)).astype(np.int32)
    tr_color = np.array(cv2.merge((transformed, transformed, transformed)), np.uint8)
    red = np.zeros_like(tr_color)
    cv2.polylines(tr_color, [transf_pts], True, (255, 0, 0), 3)
    output = cv2.addWeighted(tr_color,1.0, red, 0.5, 0.0)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(t, cmap='gray')
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(output)
    ax2.set_title(' Transformed result', fontsize=50)
    plt.show()
    
if __name__ == "__main__":
    if sys.argv[2] == 'color':
        test_perspective_color(sys.argv[1])
    else:
        test_perspective_threshold(sys.argv[1])
