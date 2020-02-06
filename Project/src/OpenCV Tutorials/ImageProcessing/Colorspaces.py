## imports
import cv2
import numpy as np

## lost the avaliable color space conversions
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)

## Object tracking using colour
cap = cv2.VideoCapture(0)

while(True):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('frame', frame)

    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    blueMask = cv2.inRange(hsvFrame, lower_blue, upper_blue)
    cv2.imshow('Blue Mask', blueMask)

    # Define range of red color in HSV
    lower_red = np.array([30, 50, 50])
    upper_red = np.array([0, 255, 255])

    redMask = cv2.inRange(hsvFrame, lower_red, upper_red)
    cv2.imshow('Red Mask', redMask)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=blueMask)
    cv2.imshow('res', res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

## Find HSV values
red = np.uint8([[[0, 0, 255]]])
hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
print(hsv_red)