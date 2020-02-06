## imports
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

## Smooth image with a 5x5 low pass kernel
img = cv2.imread('..\\..\\images\\input\\soduku.jpeg')

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

## Averaging - same as above (easier implementation)
blur = cv2.blur(img,(5,5))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred - Average')
plt.xticks([]), plt.yticks([])
plt.show()

## Gaussian Filtering - values in kernal have a normal distribution (not all weighted equally)
blur = cv2.GaussianBlur(img,(5,5),0)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred - Gaussian')
plt.xticks([]), plt.yticks([])
plt.show()

## Median Filtering - computes all pixels in kernal window, replaces centre pixel with median of those (good for removing salt-n-pepper noise)
img = cv2.imread('..\\..\\images\\input\\soduku.jpeg', 0)
noise_img = sp_noise(img, 0.05)

median = cv2.medianBlur(noise_img, 5)

plt.subplot(121),plt.imshow(noise_img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Removed Noise - Median')
plt.xticks([]), plt.yticks([])
plt.show()

## Bilateral Filtering - wont blur edges!
blur = cv2.bilateralFilter(img,9,75,75)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred - Bilateral')
plt.xticks([]), plt.yticks([])
plt.show()