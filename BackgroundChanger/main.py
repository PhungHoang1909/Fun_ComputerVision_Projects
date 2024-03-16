import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
#cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
#fpsReader = cvzone.fps()
imgBG = cv2.imread("BackgroundChanger/backgrounds/img1.jpg")

# Make sure the image are the same width and height that we defined: 640 x 480
listImg = os.listdir("BackgroundChanger/backgrounds")
imgList = []
for imPath in listImg:
    img = cv2.imread(f'BackgroundChanger/backgrounds/{imPath}')
    imgList.append(img)

imgIndex = 0

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[imgIndex], cutThreshold=0.8)

    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    #_, imgStacked = fpsReader.update(imgStacked)

    # print(imgIndex)

    #cv2.imshow("Image", imgStacked)
    cv2.imshow("Image", imgOut)
    key = cv2.waitKey(1)
    if key == ord('s'):
        if imgIndex > 0:
            imgIndex -= 1
    elif key == ord('w'):
        if imgIndex < len(imgList) - 1:
            imgIndex += 1
    elif key == ord('q'):
        break