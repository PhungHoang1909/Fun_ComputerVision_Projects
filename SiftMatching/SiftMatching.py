# OpenCV - Feature Matching Using Brute-Force Matching 

# import library
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read 2 images in Gray Scale mode
img1 = cv.imread("box.png", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("box_in_scene.png", cv.IMREAD_GRAYSCALE)

# Initiate SIFT detector
sift = cv.SIFT_create()
sift.setContrastThreshold(0.03)
sift.setEdgeThreshold(5)

# Apply SIFT to find the keypoints and descriptors with detectAndCompute 
# return 2 array of keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None) 
kp2, des2 = sift.detectAndCompute(img2, None)

# Perform Feature Matching with BFMatcher between the descriptors of the two images
# returns a list of the k nearest neighbors for each descriptor in the query set (des1) from the training set (des2). 
# k = 2 -> return top 2 matches
bf = cv.BFMatcher()
matches = bf.match(des1, des2)
matches = sorted(matches, key= lambda x:x.distance)
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], img2, flags=2)

matches = bf.knnMatch(des1, des2, k=2) 

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])
        
# cv.drawMatchesKnn expects list of lists as matches.
img4 = cv.drawMatchesKnn(
    img1, kp1, img2, kp2, good, None, matchColor=(0, 255, 0) , matchesMask=None, singlePointColor=(0, 0, 255) ,flags=0
)
plt.imshow(img4), plt.show()
