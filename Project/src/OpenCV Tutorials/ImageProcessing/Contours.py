## Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

## finding the contours of a binary image
img = cv2.imread('..\\..\\images\\input\\sidewall6.png', 0)
edges = cv2.Canny(img, 100, 200)

ret, thresh = cv2.threshold(img,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imshow('contours', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

## moments
image, contours, hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)

# Find centeroid of cnt
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

print(str(cx) + ' x ' + str(cy))

# Find area of cnt
area = cv2.contourArea(cnt)

# Find the perimeter of cnt
perimeter = cv2.arcLength(cnt, True)

img = cv2.imread('..\\..\\images\\input\\sidewall6.png')
img = cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1, cv2.LINE_AA)

cv2.imshow('centeroid', img)

cv2.waitKey(0)
cv2.destroyAllWindows()