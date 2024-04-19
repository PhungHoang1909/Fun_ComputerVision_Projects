import cv2
import numpy as np
import matplotlib.pyplot as plt


def featureMatching(img1, img2, ratio=0.75, show=False):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    sift.setContrastThreshold(0.03)
    sift.setEdgeThreshold(5)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des2, des1, k=2)

    matchesMask_ratio = [[0, 0] for i in range(len(matches))]
    match_dict = {}
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            matchesMask_ratio[i] = [1, 0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1, des2, k=2)
    matchesMask_ratio_recip = [[0, 0] for i in range(len(recip_matches))]

    for i, (m, n) in enumerate(recip_matches):
        if m.distance < ratio * n.distance:  # ratio
            if (
                m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx
            ):  # reciprocal
                good.append(m)
                matchesMask_ratio_recip[i] = [1, 0]
    if show:
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matchesMask_ratio_recip,
            flags=0,
        )
        img3 = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, recip_matches, None, **draw_params
        )

        plt.figure(), plt.xticks([]), plt.yticks([])
        plt.imshow(
            img3,
        )
        plt.savefig("feature_matching.png", bbox_inches="tight")

    return ([kp1[m.queryIdx].pt for m in good], [kp2[m.trainIdx].pt for m in good])


def getTransform(src, dst, method="affine"):
    pts1, pts2 = featureMatching(src, dst)

    src_pts = np.float32(pts1).reshape(-1, 1, 2)
    dst_pts = np.float32(pts2).reshape(-1, 1, 2)

    if method == "affine":
        M, mask = cv2.estimateAffine2D(
            src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0
        )
        M = np.append(M, [[0, 0, 1]], axis=0)

    if method == "homography":
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)


def Perspective_warping(img1, img2, img3):
    # Extend borders of img1
    img1_padded = cv2.copyMakeBorder(img1, 50, 50, 100, 500, cv2.BORDER_CONSTANT)

    # Compute transformation matrices
    (M1, _, _, _) = getTransform(img2, img1_padded, "homography")
    (M2, _, _, _) = getTransform(img3, img1_padded, "homography")

    # Warp images
    out1 = cv2.warpPerspective(img2, M1, (img1_padded.shape[1], img1_padded.shape[0]))
    out2 = cv2.warpPerspective(img3, M2, (img1_padded.shape[1], img1_padded.shape[0]))

    # Combine warped images using weighted addition
    alpha = 0.5
    beta = 1.0 - alpha
    output = cv2.addWeighted(out1, alpha, out2, beta, 0.0)

    # Display images
    plt.figure(figsize=(12, 8))

    # Original Images
    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap="gray")
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap="gray")
    plt.title("Image 2")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(img3, cmap="gray")
    plt.title("Image 3")
    plt.axis("off")

    # Resulting stitched image
    plt.subplot(2, 2, 4)
    plt.imshow(output, cmap="gray")
    plt.title("Stitched Image")
    plt.axis("off")

    plt.show()

    return True


img1 = cv2.imread("7_1.jpg")
img2 = cv2.imread("7_2.jpg")
img3 = cv2.imread("7_3.jpg")
Perspective_warping(img1, img2, img3)
