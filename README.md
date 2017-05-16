# ** Advanced Lane Finding Project **

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[image1]: ./output_images/camera_calibration.png "Undistorted"
[image2]: ./test_images/test6.jpg "Original"
[image3]: ./output_images/undistorted/test6.jpg "Corrected"
[image4]: ./output_images/threshold/test3.jpg "Thresholds Example"
[image5]: ./output_images/perspective_transformation-straight.png "Perspective Transformation"
[image6]: ./output_images/perspective_transformation-curved.png "Binary Perspective Transformation"
[image7]: ./output_images/sliding-window-1.png "Sliding Window"
[image8]: ./output_images/sliding-window-2.png "Find Lines"
[image9]: ./output_images/lanes/test6.jpg "Draw Lines"
[video]: ./video.mp4 "Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function `calibrate` implemented by lines 7 through 40 of the file called `calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Here is an example of applying correction to on of the test images:

| Original image | Corrected image |
|:--------------:|:-----------------:|
|![alt text][image2]| ![alt text][image3]|

All of the corrected test images can bi found in the folder `output_images/undistorted`. The corrections is implemented in the processing pipeline code line 35 of file `pipeline.py`.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 7 through 61 in `threshold.py`).  The transformation is implemented in function `threshold` (lines 43 - 61). As a first step I converted the image to HLS color space and used L channel for gradient thresholds: x, y, direction and magnitude. S and H channels I used for color thresholds. Finally I combined all thresholds to calculate the binary output (line 59). Here's an example of my output for this step.  (other examples are stored in the folder `output_images/threshold`)

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes tow functions: `perspective()` and `transform()`defined in the file `perspective.py`  The `perspective()` function takes as inputs source and destination points defining the transformation and returns transformation matrixes. The `transform()` function applies the transformation to an image. This function is called per each frame in the processing pipeline. I manually picked the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 253, 686      | 253, 686        |
| 1053, 686      | 1053, 686      |
| 702, 460     | 1053, 0      |
| 582, 460      | 253, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image:
![alt text][image5]
And here is an example of binary warped image (other test images are available in folder `output_images/warped`):
![alt text][image6]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used a sliding window approach to identify lane-line pixels. The implementation is provided in file `lanes.py`. The main function called from the processing pipeline is `sliding_window()` (code lines 181 to 216). It uses a pair of Line objects for left and right line. The class Line (code lines 14 to 167) holds the current state of detection as well as the actual sliding window implementation. The two main processing methods of the class are `detect()` and `find_lines()`. The method `detect()` (code lines 47 to 92) is invoked for initial detection and uses sliding window to find the line pixels.
Here is an example of the result from `detect()` call:
![alt text][image7]
The `find_lines()` is used when there is a fitted polynomial from previous frames and it uses four times smaller margin.
![alt text][image8]

The class Line holds the results for the last 10 frames and produces a weighted average over them for smoothing the results (lines 150 to 157).

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of the lane and the position of the vehicle with respect to center is calculated for the current frame in lines 128 through 134 in the file  `lanes.py` The values for the current frame are then stored in Line objects (lines 146 and 147) if the detection is not rejected and the best fit is averaged in lines 153 and 154.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 219 through 243 in my code in `lanes.py` in the function `draw_lane()`.  Here is an example of my result on a test image (other images are in folder `output_images/lanes`):

![alt text][image9]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The video processing pipeline is defined in file `pipeline.py`. Each frame is processed by function `pipeline()` implemented in code lines 19 through 39. It is wrapped into a helper function `process_image()` which is a callback for video stream.
Here's a [link to my video result](./video.mp4)

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major difficulty I faced is the selection of the right combination of thresholds for producing the binary image. Some combinations worked better on one set of images but poorly on others. There is a need for more experiments to achieve a better performance. Also the sliding window approach is very sensitive to the size of the window. I did not try to use convolutional method for pixel detection and can not comment on which choice is better.

Other issue I found is the curvature variance. My attempt to use the difference in curvature as rejections criteria failed with too high rejection rate.

Despite these issues I enjoyed working on it.  
