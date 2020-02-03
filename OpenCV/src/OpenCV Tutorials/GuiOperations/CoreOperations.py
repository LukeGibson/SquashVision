## imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

## load image and print pixel value
img = cv2.imread('..\\images\\input\\sidewall6.png')
pxArray = img[100, 100]
pxBlue = img.item(100, 100, 0)

print(pxArray)

## Set the value of a pixel
img[100, 100] = [0, 0, 255]
img.itemset((100, 100, 0), 255)

print(img[100, 100])

## Get image info
print(img.shape)  #(Row, Col, Chamnel)
print('contains ' + str(img.size) + ' pixels')
print('datatpye: ' + str(img.dtype))

## Copy a region of an image
ball = img[450:550, 1000:1200]
img[700:800, 1000:1200] = ball

cv2.imshow('squash ball detector', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Split the channels of an image
b,g,r = cv2.split(img)
cv2.imshow('blue channel', b)
cv2.waitKey(0)
cv2.imshow('green channel', g)
cv2.waitKey(0)
cv2.imshow('red channel', r)
cv2.waitKey(0)

img = cv2.merge((b,g,r))
cv2.imshow('merged channels', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Set all of one channel to 0
img[:, :, 2] = 0
cv2.imshow('red channel set to 0', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Padding images
BLUE = [255,0,0]

img1 = cv2.imread('..\\images\\input\\sidewall6.png')

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()

## Adding 2 images (must be equal depth and type, or 2nd image be a scalar)
img1 = cv2.imread('..\\images\\input\\sidewall6.png')
img2 = cv2.imread('..\\images\\input\\sidewall1.png')

res = cv2.add(img1, img2)

cv2.imshow('added images', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Image blending - same as adding, but iamges are given differnet weights
dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

cv2.imshow('blended images', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Bitwise operations (AND OR NOT XOR)
# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

# Now create a mask of img2 and create its inverse mask also
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of img2 in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of img2 from img2.
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
