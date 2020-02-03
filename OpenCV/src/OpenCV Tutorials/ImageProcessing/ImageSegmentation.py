## imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

## Find approximate estimate for the coins - using otus's binerization
img = cv2.imread('..\\..\\images\\input\\coins.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow('threshold', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Remove noise with opening
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

cv2.imshow('noise removed', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Use erosion and dilation to find the 'sure' backfround and foregrounds
sure_bg = cv2.dilate(opening,kernel,iterations=3)

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

cv2.imshow('sure foreground (white)', sure_fg)
cv2.imshow('sure background (black)', sure_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Find the unknown region as the difference between the 2
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

cv2.imshow('unknown region (white)', unknown)
cv2.waitKey(0)
cv2.destroyAllWindows()

## label the regions with a marker
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

#apply watershed algorithm
markers = cv2.watershed(img, markers)
img[markers == -1] = [255,0,0]

cv2.imshow('result', ret)
cv2.waitKey(0)
cv2.destroyAllWindows()