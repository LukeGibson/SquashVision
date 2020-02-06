## Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

## calculate a histogram from grayscale pixel intesnity
img = cv2.imread('..\\..\\images\\input\\sidewall6.png', 0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

## plot histogram of grayscale
plt.hist(img.ravel(), 256, [0, 256])
plt.show()

## plot bgr histogram
img = cv2.imread('..\\..\\images\\input\\sidewall6.png')

color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

## Histogram equalization - to imporve contrast
img = cv2.imread('..\\..\\images\\input\\lowcontrast.png', 0)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

## Find the minimal histogram value (excluding 0)
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

img2 = cdf[img]

hist, bins = np.histogram(img2.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(img2.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(img2), plt.title('Output')
plt.show()

## Histogram equalisation with openCV
img = cv2.imread('..\\..\\images\\input\\lowcontrast.png', 0)
equ = cv2.equalizeHist(img)

res = np.hstack((img, equ))  # stacking images side-by-side
plt.imshow(res), plt.title('OpenCV Histogram Equalisation')
plt.show()

## CLAHE equalisation - when you're histogram isn't bunch up (both light and dark pixles present)
img = cv2.imread('..\\..\\images\\input\\CLAHE.png', 0)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
res = np.hstack((img, cl1))  # stacking images side-by-side

plt.imshow(res), plt.title('OpenCV CLAHE Histogram Equalisation')
plt.show()

## 2D histograms - using hsv instead of grayscale gives us 2D (Hue = 0-180 and Saturation = 0-256)
img = cv2.imread('..\\..\\images\\input\\sidewall6.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

plt.subplot(221), plt.imshow(img), plt.title('BRG Image')
plt.subplot(222), plt.imshow(hsv), plt.title('HSV Image')
plt.subplot(223), plt.imshow(hist, interpolation='nearest'), plt.title('2D Histogram')
plt.show()

## Histogram back projection - shows how likly a pixel is to be an object
roi = cv2.imread('..\\..\\images\\input\\wall.png')
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

target = cv2.imread('..\\..\\images\\input\\sidewall6.png')
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

# calculating object histogram
roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# normalize histogram and apply backprojection
cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)

# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cv2.filter2D(dst, -1, disc, dst)

# threshold and binary AND
ret, thresh = cv2.threshold(dst, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))
res = cv2.bitwise_and(target, thresh)

res = np.vstack((target, res))
plt.imshow(res)
plt.show()

# cv2.imshow('matched pixles', thresh)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
