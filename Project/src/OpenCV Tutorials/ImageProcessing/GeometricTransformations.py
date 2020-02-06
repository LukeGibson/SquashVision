## imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

## Scaling
img = cv2.imread('..\\..\\images\\input\\sidewall6.png', 0)
res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# OR
height, width = img.shape[:2]
res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)

# images = [img, res]
# titles = ['original', 'resized']
#
# for i in range(2):
#     plt.subplot(1, 2, i+1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

cv2.imshow('original', img)
cv2.imshow('resize', res)

cv2.waitKey(0)
cv2.destroyAllWindows()

## Translation
rows, cols = img.shape

# create a numpy 2x3 matrix
matrix = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img, matrix, (cols + 100, rows + 50))

cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Rotation
matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
dst = cv2.warpAffine(img, matrix, (cols, rows))

cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Affine Transform
rows, cols = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

matrix = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img,matrix,(cols, rows))

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()

## Perspective Transform
img = cv2.imread('..\\..\\images\\input\\soduku.jpeg')
rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,matrix,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()