import cv2
import numpy as np


def estimateSpeed(I1, I2, roi):
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    grayI1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    grayI2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    optical_flow, status, err = cv2.calcOpticalFlowPyrLK(
        grayI1, grayI2, roi, None, **lk_params
    )

    speedMagnitude = np.sqrt(optical_flow[0][0][0] ** 2 + optical_flow[0][0][1] ** 2)

    good_points = optical_flow[status == 1]

    roi = good_points.reshape(-1, 1, 2)

    return speedMagnitude, roi


def main():
    video = "video.mp4"
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, prev = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    x, y, w, h = cv2.selectROI(prev, False)
    roi = np.array([[[x + w / 2, y + h / 2]]], dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        speed, roi = estimateSpeed(prev, frame, roi)

        x, y = roi[0][0]
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))

        # Display the speed above the rectangle
        text_position = (top_left[0], top_left[1] - 10)  # 10 pixels above the rectangle

        cv2.putText(
            frame,
            f"Speed: {speed:.2f} pixels/frame",
            text_position,
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (0, 255, 0),
            1,
        )
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow("Object Tracking", frame)

        prev = frame.copy()
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
