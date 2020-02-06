## Imports
import cv2
import numpy as np

## Background Subtractor MOG
cap = cv2.VideoCapture('..\\..\\videos\\input\\sidewallclips.mp4')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    if ret == True:
        fgmask = fgbg.apply(frame)

        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

## Background subtrackor MOG2 - addaptive k (better for varying illumination) - picks up shadows
cap = cv2.VideoCapture('..\\..\\videos\\input\\sidewallclips.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

## Background subtractor GMG - uses first (default=120) frames for background modelling
cap = cv2.VideoCapture('..\\..\\videos\\input\\ball.mp4')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()