# Capture frames from MP4 or true camera, overlay your text and logo (png/jpg) into frame

import cv2

def process_video(capture_source):
    # Access the video source (camera or mp4 file)
    cap = cv2.VideoCapture(capture_source)

    # Load the logo image and resize it
    logo = cv2.imread('Camera_ImageFormation\student.jpg')
    logo_height, logo_width, _ = logo.shape

    # Define the desired size for the logo (you can adjust this as needed)
    desired_logo_width = 100
    desired_logo_height = int((desired_logo_width / logo_width) * logo_height)
    logo = cv2.resize(logo, (desired_logo_width, desired_logo_height))

    
    #Set the window to full screen
    cv2.namedWindow('video', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()

        # Insert text on video
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,
                    'Le Nguyen Phung Hoang - 2151013026',
                    (50, 50),
                    font, 1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)

        frame[100:100 + desired_logo_height, 50:50 + desired_logo_width] = logo

        cv2.imshow('video', frame)

        # Break the loop and exit the video when 'q' key is pressed+
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Uncomment the line below to process the laptop camera (camera index 0)
#process_video(0)

# Uncomment the line below to process the mp4 file
process_video('Camera_ImageFormation\BackgroundVideo.mp4')