## Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

## Create an edge image
img = cv2.imread('..\\..\\images\\input\\sidewall6.png', 0)
edges = cv2.Canny(img, 100, 200)

cv2.imshow('original', img)
cv2.imshow('canny-edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
