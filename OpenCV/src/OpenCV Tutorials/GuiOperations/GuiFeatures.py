## Import libaries
import numpy as np
import cv2
from matplotlib import pyplot as plt

## Load an color image in grayscale
img = cv2.imread('..\\images\\input\\sidewall6.png', cv2.IMREAD_GRAYSCALE)

## Show an image (then chose whether to save it)
cv2.imshow('squash ball detector', img)

k = cv2.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):  # wait for 's' key to save and exit
    cv2.imwrite('..\\images\\output\\sidewall6-out.png', img)
    cv2.destroyAllWindows()

## Display image in matplotlib
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

## Play video from webcam
cap = cv2.VideoCapture(0)

print(cap.get(3))
print(cap.get(4))

# Set video resolution
cap.set(3, 1920)
cap.set(4, 1080)

print(cap.get(3))
print(cap.get(4))

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

## Play video from a file
cap = cv2.VideoCapture('..\\videos\\input\\SW-FAR_CLIPPED_3_normal-balls.mp4')
cap.open('..\\videos\\input\\SW-FAR_CLIPPED_3_normal-balls.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Sidewall clips', gray)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

## Load a video, modify it, then save it
cap = cv2.VideoCapture('..\\videos\\input\\SW-FAR_CLIPPED_3_normal-balls.mp4')
cap.open('..\\videos\\input\\SW-FAR_CLIPPED_3_normal-balls.mp4')

print('Input video resolution = ' + str(cap.get(3)) + ' x ' + str(cap.get(4)))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('..\\videos\\output\\recording-flipped.mp4', fourcc, 15, (1920, 1080))

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('recording clip', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

## Drawing on an image
img = img = np.zeros((512, 512, 3), np.uint8)

# Draw various set shapes
# img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
# img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
# img = cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
# img = cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
#
# # Draw a custom polygon
# pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
# pts = pts.reshape((-1, 1, 2))
# img = cv2.polylines(img, [pts], True, (255, 255, 255))

# Creating the OpenCV logo
img = cv2.circle(img, (150, 280), 90, (0, 255, 0), -1, cv2.LINE_AA)
img = cv2.circle(img, (150, 280), 36, (0, 0, 0), -1, cv2.LINE_AA)

pts = np.array([[150, 280], [150 + 90, 280], [150 + 90, 280 - 60], [150 + 53, 280 - 90]], np.int32)
pts = pts.reshape((-1, 1, 2))
img = cv2.fillPoly(img, [pts], (0, 0, 0))

img = cv2.circle(img, ((512 - 150), 280), 90, (255, 0, 0), -1, cv2.LINE_AA)
img = cv2.circle(img, ((512 - 150), 280), 36, (0, 0, 0), -1, cv2.LINE_AA)

pts = np.array([[(512 - 150), 280], [512 - 150 - 60, 280 - 90], [512 - 150 + 60, 280 - 90]], np.int32)
pts = pts.reshape((-1, 1, 2))
img = cv2.fillPoly(img, [pts], (0, 0, 0))

img = cv2.circle(img, (256, 100), 90, (0, 0, 255), -1, cv2.LINE_AA)
img = cv2.circle(img, (256, 100), 36, (0, 0, 0), -1, cv2.LINE_AA)

pts = np.array([[256, 100], [256 - 60, 100 + 90], [256 + 60, 100 + 90]], np.int32)
pts = pts.reshape((-1, 1, 2))
img = cv2.fillPoly(img, [pts], (0, 0, 0))

# Write text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (15, 470), font, 4, (255, 255, 255), 10, cv2.LINE_AA)

cv2.imshow('drawing', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
