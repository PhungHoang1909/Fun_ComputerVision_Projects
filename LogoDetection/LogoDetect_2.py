# Logo Detection in MP4 Video
# Using Histogram Backprojection in OpenCV:
# cv2.calcBackProject( target_img, channels, roi_hist, ranges, scale )

import cv2
import numpy as np


def detect_logo(video_path, logo_path):
    logo_img = cv2.imread(logo_path)
    logo_hsv = cv2.cvtColor(logo_img, cv2.COLOR_BGR2HSV)
    # Calculate the histogram of the logo in the HSV color space
    logo_hist = cv2.calcHist([logo_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the ranges for backprojection
        ranges = [0, 180, 0, 256]  # Hue range (0-179), Saturation range (0-255)

        # Perform histogram backprojection
        dst = cv2.calcBackProject([frame_hsv], [0, 1], logo_hist, ranges, scale=1)

        ret, thresh = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        cv2.imshow("Video with Logo Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


video_path = "output_video.mp4"
logo_path = "logo.png"
detect_logo(video_path, logo_path)
