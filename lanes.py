import numpy as np
import cv2
import matplotlib.pyplot as plt
import threshold as thr
import perspective as persp
import sys
import calibration as cal

class Line():
    def __init__(self, nwindows = 18, margin = 100, minpix = 200):
        self.nwindows = nwindows
        self.margin = margin
        self.detected_margin = int(margin/2)
        self.minpix = minpix
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30.0/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension

        
    def detect(self, binary_warped, x_base):     
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/self.nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        x_current = x_base

        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        edge_reached = False

        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - self.margin
            win_x_high = x_current + self.margin

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            if not edge_reached:
                lane_inds.append(good_inds)
                if len(good_inds) > self.minpix:
                    x_current = np.int(np.mean(nonzerox[good_inds]))
                elif win_y_low < binary_warped.shape[0]/2:
                    edge_reached = True

            if x_current - self.margin <= 0 or (x_current + self.margin) >= binary_warped.shape[1]:
                    edge_reached = True
                
        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)
        
        if len(lane_inds) > 2:
            # Extract line pixel positions
            self.allx = nonzerox[lane_inds]
            self.ally = nonzeroy[lane_inds] 

            # Fit a second order polynomial to each
            self.current_fit = np.polyfit(self.ally, self.allx, 2)
            self.fitx(binary_warped)
            self.detected = True
            
    def find_lines(self, binary_warped):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        fit = self.current_fit
    
        lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - self.detected_margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + self.detected_margin))) 

        if len(lane_inds) > 2:
            # Again, extract left and right line pixel positions
            # Extract left and right line pixel positions
            self.allx = nonzerox[lane_inds]
            self.ally = nonzeroy[lane_inds] 

            # Fit a second order polynomial to each
            self.current_fit = np.polyfit(self.ally, self.allx, 2)
            self.fitx(binary_warped)
        else:
            self.detected = False

    def fitx(self, binary_warped):
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        self.recent_xfitted = self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]
        self.ploty = ploty
        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty*self.ym_per_pix, self.recent_xfitted*self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*self.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])


def sliding_window(binary_warped, left_line, right_line):
    """
    Parameters:
    - binary_warped - bynary warped image
    - nwindows - number of windows
    - margin - width of the windows +/- margin
    - minpix - minimum number of pixels found to recenter window

    Returns left and right polynomial coefficients
    """
    
    if left_line.detected and right_line.detected:
        left_line.find_lines(binary_warped)
        right_line.find_lines(binary_warped)
    else:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]*3/4):,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        left_line.detect(binary_warped, leftx_base)
        right_line.detect(binary_warped, rightx_base)

def draw_lane(binary_warped, left_line, right_line, Minv, orig_img):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.recent_xfitted, left_line.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.recent_xfitted, right_line.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (orig_img.shape[1], orig_img.shape[0])) 
    # Combine the result with the original image
    return cv2.addWeighted(orig_img, 1, newwarp, 0.3, 0)
    
def test_sliding_window(img_path):
    margin = 100
    mtx, dist =cal.calibrate('camera_cal/', 9,  6)
    img = plt.imread(img_path)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx) 
    t = thr.threshold(undistorted)
    M, _ = persp.perspective()
    binary_warped = persp.transform(t, M)
    left_line = Line()
    right_line = Line()
    sliding_window(binary_warped, left_line, right_line)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_line.current_fit[0]*ploty**2 + left_line.current_fit[1]*ploty + left_line.current_fit[2]
    right_fitx = right_line.current_fit[0]*ploty**2 + right_line.current_fit[1]*ploty + right_line.current_fit[2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))* 255.999) .astype(np.uint8)
    out_img[left_line.ally, left_line.allx] = [255, 0, 0]
    out_img[right_line.ally, right_line.allx] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

    sliding_window(binary_warped, left_line, right_line)
    
    # Generate x and y values for plotting
    left_fitx = left_line.current_fit[0]*ploty**2 + left_line.current_fit[1]*ploty + left_line.current_fit[2]
    right_fitx = right_line.current_fit[0]*ploty**2 + right_line.current_fit[1]*ploty + right_line.current_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))* 255.999) .astype(np.uint8)
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[left_line.ally, left_line.allx] = [255, 0, 0]
    out_img[right_line.ally, right_line.allx] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-left_line.detected_margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+left_line.detected_margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-right_line.detected_margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+right_line.detected_margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()
    

if __name__ == "__main__":
    test_sliding_window(sys.argv[1])
