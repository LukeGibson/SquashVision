## Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sys import exit

## Load a video
def loadCapture():
    footagePath = 'videos\\project\\input\\'
    videoName = 'clipout.mp4'
    capture = cv2.VideoCapture(footagePath + videoName)

    if not capture.isOpened():
        print("Can't open video:" + videoName)
        exit(0)
    else:
        return capture

## Create different background subtractors
subtractorMOG = cv2.bgsegm.createBackgroundSubtractorMOG()  # Gaussian Based
subtractorCNT = cv2.bgsegm.createBackgroundSubtractorCNT()  # Based on counting (2x faster than MOG2)
subtractorMOG2 = cv2.createBackgroundSubtractorMOG2()  # Adaptive MOG (works better with varying illumination)
subtractorKNN = cv2.createBackgroundSubtractorKNN()  # Uses KNN (very efficent if number of foreground pixels is very low)

subtractors = [subtractorMOG, subtractorCNT, subtractorMOG2, subtractorKNN]
subtractorNames = ['MOG', 'CNT', 'MOG2', 'KNN']

## Extract foreground
for x in range(len(subtractorNames)):
    subtractor = subtractors[x]
    subtractorName = subtractorNames[x]

    capture = loadCapture()

    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'MPV4')
    output = cv2.VideoWriter('videos\\project\\output\\FgExtraction\\BgSubtractor\\' + subtractorName + '.mp4', fourcc, 20.0, (1920, 1080))

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create binary mask using subtractor
        fgMask = subtractor.apply(frameGray)

        # Perform morphological opening (erosion followed by dilation) - to remove noise from mask
        kernel = np.ones((5, 5), np.uint8)
        fgMaskOpening = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

        # overlay mask on original image
        # overlay = cv2.add(frameGray, fgMaskOpening)

        # stamp frame with frame number and subtractor type
        cv2.rectangle(fgMaskOpening, (5, 5), (100, 40), (255, 255, 255), -1)
        cv2.putText(fgMaskOpening, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.putText(fgMaskOpening, subtractorName, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        output.write(fgMaskOpening)

        cv2.imshow('Extracted Foreground', fgMaskOpening)

        key = cv2.waitKey(25) & 0xff
        if key == 27:
            break

    capture.release()
    output.release()
    cv2.destroyAllWindows()
