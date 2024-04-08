import cv2
import numpy as np

img = cv2.imread('source2.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for the  image
keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)

# Initialize FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for the current frame
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray, None)

    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    good = []
    for m, n in matches:
        if m.distance <= 0.9 * n.distance:
            good.append(m)

    img3 = cv2.drawMatches(img, keypoints_1, gray, keypoints_2, good, None, matchColor=(0, 255, 0), flags=0)

    cv2.imshow('Flann Match', img3)

    if len(good) >= 10:
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()