import cv2
import calibration as cal
import threshold as th
import perspective as persp
import lanes
from moviepy.editor import VideoFileClip
import sys

#Parameters:
c_thresh=(170, 255)
s_thresh=(20, 100)
m_thresh=(20, 100)
dir_thresh=(0.7, 2.4)
h_thresh=(0,130)
nwindows = 18
margin = 80
minpix = 200

def pipeline(img, left_line, right_line, mtx, dist, pmtx, pmtx_inv):
    """
    Main processing function.

    Parameters:
    `img` - current video frame
    `left_line` and `right_line` - Line objects holding currently detected
    left and right lines.
    `mtx` - calibration matrix
    `dist` - calibration coefficients
    `pmtx` - perspective transformation matrix
    `pmtx_inv` - inverse perspective transformation matrix

    Returns an image with detected lane drawn.
    """

    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    binary = th.threshold(undistorted, c_thresh=c_thresh, s_thresh=s_thresh, m_thresh=m_thresh, dir_thresh=dir_thresh, h_thresh=h_thresh)
    binary_warped = persp.transform(binary, pmtx)
    lanes.sliding_window(binary_warped, left_line, right_line)
    return lanes.draw_lane(binary_warped, left_line, right_line, pmtx_inv, undistorted)

def generate_process_image():
    """
    Generate a wrapper function around the pipeline suitable for
    video clip processing.
    Return finction of one argument - image.
    """
    mtx, dist = cal.calibrate('camera_cal/', 9,  6)
    M,Minv = persp.perspective()
    left_line = lanes.Line(nwindows =nwindows, margin = margin)
    right_line = lanes.Line(nwindows =nwindows, margin = margin)

    def process_image(image):
        return pipeline(image, left_line, right_line, mtx, dist, M, Minv)

    return process_image

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <input video clip> <output video clip>\n")
        exit(1)

    output = sys.argv[2]
    clip = VideoFileClip(sys.argv[1])
    processed_clip = clip.fl_image(generate_process_image())
    processed_clip.write_videofile(output, audio=False)
