import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Threshold the gradient
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary

def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    scaled = np.uint8(255 * magnitude / np.max(magnitude))
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary_output

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    direction = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return binary_output

def color_threshold(s_channel, thresh=(170,255)):
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary

def threshold(img, ksize = 3,  c_thresh=(170, 255), s_thresh=(20, 100), m_thresh=(20, 100), dir_thresh=(0.7, 2.4), h_thresh=(0,130)):
    img = np.copy(img)
    # Convert to HLS color space and separate the L channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    gradx = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=ksize, thresh=s_thresh)
    grady = abs_sobel_thresh(l_channel, orient='y', sobel_kernel=ksize, thresh=s_thresh)
    mag_binary = mag_thresh(l_channel, sobel_kernel=ksize, thresh=m_thresh)
    dir_binary = dir_threshold(l_channel, sobel_kernel=ksize, thresh=dir_thresh)
    color_s_binary = color_threshold(s_channel, thresh=c_thresh)
    color_h_binary = color_threshold(h_channel, thresh=h_thresh)

    combined = np.zeros_like(l_channel)
    combined[(((gradx == 1) & (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1) & (color_h_binary == 1))) | ((color_s_binary == 1) )] = 1

     # Stack each channel
    return combined

def test_threshold(img_path):
    img = plt.imread(img_path)
    binary = threshold(img)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(binary)
    ax2.set_title(' Binary result', fontsize=50)
    plt.show()
    
if __name__ == "__main__":
    test_threshold(sys.argv[1])
