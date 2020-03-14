## imports
import cv2
import numpy as np
from sys import exit


## Load origial video
def loadOriginalCapture():
    footagePath = 'videos\\project\\input\\'
    videoName = 'clipout.mp4'
    capture = cv2.VideoCapture(footagePath + videoName)

    if not capture.isOpened():
        print("Can't open video:" + videoName)
        exit(0)
    else:
        return capture


# Load the original video
originalCapture = loadOriginalCapture()

# Create output writer
fourcc = cv2.VideoWriter_fourcc(*'MPV4')
output = cv2.VideoWriter('videos\\project\\output\\LineDetection\\CannyEdges\\' + 'clipout.mp4', fourcc, 20.0, (1920, 1080))

while True:
    originalRet, origialFrame = originalCapture.read()

    if not originalRet:
        break

    origialFrameGray = cv2.cvtColor(origialFrame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(origialFrameGray, 200, 250)

    output.write(edges)

    cv2.imshow('Edges', edges)

    key = cv2.waitKey(25) & 0xff
    if key == 27:
        break

output.release()
cv2.destroyAllWindows()

