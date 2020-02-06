## Imports
import cv2
import numpy as np

## Erosion
img = cv2.imread('..\\..\\images\\input\\j.png', 0)
height, width = img.shape[:2]

img = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)

kernel = np.ones((10,10),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

cv2.imshow('erosion', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Dilation
dilation = cv2.dilate(img, kernel, iterations = 1)

cv2.imshow('dilation', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Opening - erosion followed by dilation (removes noise)
img = cv2.imread('..\\..\\images\\input\\j-noise.png', 0)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow('opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Closing - dilation followed by closing (fills in small holes)
img = cv2.imread('..\\..\\images\\input\\j-holes.png', 0)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closing', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Gradient - the difference between dilation and erosion
img = cv2.imread('..\\..\\images\\input\\j.png', 0)
height, width = img.shape[:2]

img = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('gradient', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Structuring Element - for when you want a more complex kernal that all 1's or 0's
elipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
print(elipse)

rectangle = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
print(rectangle)

cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
print(elipse)
