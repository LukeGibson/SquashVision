from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from sys import exit
import cv2
import numpy as np


# Shared Operations


def getResizeDim(image):
    global showA, showB

    currentDisplay = displayA

    if not showA and not showB:
        print("No display showing")
        return image
    elif not showA:
        currentDisplay = displayB


    # get frame dimension
    frameWidth = currentDisplay.winfo_width()
    frameHeight = currentDisplay.winfo_height()

    # calculate scales required to fit image in frame
    heightScale = int((frameHeight / image.shape[0]) * 100)
    widthScale = int((frameWidth / image.shape[1]) * 100)

    # select most restricting scale
    scale = heightScale
    if widthScale < heightScale:
        scale = widthScale

    # scale original image
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    resizeDim = (width, height)

    return resizeDim

def showDisplayA():
    global showA, root, showInputBut

    showA = not showA
    if showA:
        showInputBut.configure(text="Hide Input Display")
    else:
        showInputBut.configure(text="Show Input Display")

    changeDisplay()


def showDisplayB():
    global showB, root, showOutputBut

    showB = not showB
    if showB:
        showOutputBut.configure(text="Hide Output Display")
    else:
        showOutputBut.configure(text="Show Output Display")

    changeDisplay()


def changeDisplay():
    global showA, showB, displayA, displayB, panelA, panelB

    if displayA is not None:
        displayA.destroy()
        panelA.destroy()

    if displayB is not None:
        displayB.destroy()
        panelB.destroy()

    displayA = Frame(root, bg="#cccccc")
    displayB = Frame(root, bg="#cccccc")

    if showA and showB:
        displayA.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.01)
        displayB.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.51)
    elif showA:
        displayA.place(relwidth=0.75, relheight=0.98, relx=0.01, rely=0.01)
    elif showB:
        displayB.place(relwidth=0.75, relheight=0.98, relx=0.01, rely=0.01)

    panelA = Label(displayA)
    panelA.pack()
    panelB = Label(displayB)
    panelB.pack()


# Image Operations


def showImage(cv2Image, isA):
    global panelA, panelB

    resizeDim = getResizeDim(cv2Image)
    image = cv2.resize(cv2Image, resizeDim, interpolation=cv2.INTER_AREA)

    # convert scale to tkinter format
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    if isA:
        # add image to display A
        panelA.configure(image=image)
        panelA.image = image
    else:
        # add image to display B
        panelB.configure(image=image)
        panelB.image = image


def selectImage():
    global currImgFile

    # open a file selection box
    filename = filedialog.askopenfilename(title="Select Image")
    currImgFile = filename

    if len(currImgFile) > 0:
        image = cv2.imread(currImgFile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        showImage(image, True)


def cannyEdgeImg():
    global currImgFile

    if len(currImgFile) > 0:
        # load image
        image = cv2.imread(currImgFile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.Canny(image, 200, 250)

        showImage(image, False)


# not really working
def redMaskImg():
    global currImgFile

    if len(currImgFile) > 0:
        # load image
        image = cv2.imread(currImgFile)
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        redUpper = (255, 200, 200)
        redLower = (40, 10, 10)

        mask = cv2.inRange(hsv, redLower, redUpper)
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)

        showImage(mask, False)


def operateImg():
    global currImgOp

    if currImgOp.get() == "CannyEdge Detection":
        cannyEdgeImg()
    if currImgOp.get() == "Red Mask":
        redMaskImg()


# Video


def selectVideo():
    global currVidFile, currCapture, pause

    # open a file selection box
    filename = filedialog.askopenfilename(title="Select Video")
    currVidFile = filename

    # check file was selected and set currCapture
    if len(currVidFile) > 0:
        print("Selected: ", currVidFile)
        currCapture = cv2.VideoCapture(currVidFile)

        # Check capture opened and load display first frame
        if not currCapture.isOpened():
            print("Can't open video: ", currVidFile)
        else:
            ret, frame = currCapture.read()

            if ret:
                pause = False
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                showImage(image, True)

            currCapture.release()


def operateVid():
    global currVidFile, currCapture, trackList

    # Release capture and create new one capture
    currCapture = cv2.VideoCapture(currVidFile)

    # Reset trackList
    trackList = []

    # Check capture opened and load display first frame
    if not currCapture.isOpened():
        print("Can't open video: ", currVidFile)
    else:
        playVideo()


def playPause():
    global pause, pauseBut

    pause = not pause
    if pause:
        pauseBut.configure(text="I>")
    else:
        pauseBut.configure(text="||")


def playVideo():
    global panelA, panelB, root, currCapture, currVidOp, pause, bgSubMOG, bgSubMOG2, trackList

    if not pause:
        ret, frame = currCapture.read()

        if not ret:
            currCapture.release()
            print("Video End")
        else:
            resizeDim = getResizeDim(frame)
            operatedImage = frame

            # perform operation
            if currVidOp.get() == 'CannyEdge Detection':
                operatedImage = cannyEdgeVid(frame)
            elif currVidOp.get() == 'BackgroundSub MOG':
                operatedImage = backgroundSubVid(frame, bgSubMOG)
            elif currVidOp.get() == 'BackgroundSub MOG2':
                operatedImage = backgroundSubVid(frame, bgSubMOG2)
            elif currVidOp.get() == 'Draw Ball Outline':
                operatedImage = drawBallVid(frame, bgSubMOG)
            elif currVidOp.get() == 'Track Ball Flight':
                operatedImage = trackBallVid(frame, bgSubMOG, trackList)


            # resize
            image = cv2.resize(frame, resizeDim, interpolation=cv2.INTER_AREA)
            operatedImage = cv2.resize(operatedImage, resizeDim, interpolation=cv2.INTER_AREA)

            # convert format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image=image)

            operatedImage = Image.fromarray(operatedImage)
            operatedImage = ImageTk.PhotoImage(image=operatedImage)

            # display original frame
            panelA.image = image
            panelA.configure(image=image)

            # display operated frame
            panelB.image = operatedImage
            panelB.configure(image=operatedImage)

            root.after(5, playVideo)
    else:
        root.after(500, playVideo)


def cannyEdgeVid(frame):
    # perform edge detection
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameEdges = cv2.Canny(frameGray, 200, 250)

    return frameEdges


def backgroundSubVid(frame, subtractor):
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create binary mask using subtractor
    mask = subtractor.apply(frameGray)

    # Perform morphological opening (erosion followed by dilation) - to remove noise from mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def drawBallVid(frame, subtractor):
    # blur and convert to grayscale
    frameBlurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frameGray = cv2.cvtColor(frameBlurred, cv2.COLOR_BGR2GRAY)

    # Create binary mask using subtractor
    mask = subtractor.apply(frameGray)

    # Perform morphological opening (erosion followed by dilation) - to remove noise from mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # find contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # generate output frame
    outputFrame = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)

    if len(contours) > 0:
        largestCon = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largestCon)

        M = cv2.moments(largestCon)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        cv2.circle(outputFrame, (int(x), int(y)), int(radius), (0, 255, 0), 1)
        cv2.circle(outputFrame, center, 1, (255, 0, 0), -1)

    return outputFrame


def trackBallVid(frame, subtractor, trackList):
    # blur and convert to grayscale
    frameBlurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frameGray = cv2.cvtColor(frameBlurred, cv2.COLOR_BGR2GRAY)

    # Create binary mask using subtractor
    mask = subtractor.apply(frameGray)

    # Perform morphological opening (erosion followed by dilation) - to remove noise from mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # find contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # generate output frame
    outputFrame = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)

    if len(contours) > 0:
        largestCon = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largestCon)

        cv2.circle(outputFrame, (int(x), int(y)), int(radius), (0, 255, 0), 1)
        trackList.append((int(x), int(y)))

    for point1, point2 in zip(trackList, trackList[1:]): 
        cv2.line(outputFrame, point1, point2, [255, 0, 0], 2) 

    return outputFrame


# Initialise App
root = Tk()

# Global Variables
currImgFile = ""
currImgOp = StringVar()
currImgOp.set("CannyEdge Detection")

currVidFile = ""
currVidOp = StringVar()
currVidOp.set("CannyEdge Detection")
currCapture = cv2.VideoCapture(0)
pause = False

bgSubMOG = cv2.bgsegm.createBackgroundSubtractorMOG()
bgSubMOG2 = cv2.createBackgroundSubtractorMOG2()
trackList = []

showA = True
showB = True

# calculate relative app dimensions for screen resolution
screenHeight = root.winfo_screenheight()
screenWidth = root.winfo_screenwidth()

rHeight = round(screenHeight/1.2)
rWidth = round(screenWidth/1.8)
rDim = str(rWidth) + "x" + str(rHeight)

root.geometry(rDim)
root.title("Operator Tester")

# create container frames
displayA = Frame(root, bg="#cccccc")
displayB = Frame(root, bg="#cccccc")

displayA.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.01)
displayB.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.51)

displayC = Frame(root)
displayC.place(relwidth=0.23, relheight=0.98, relx=0.76, rely=0.01)

# initialise labels to hold images
panelA = Label(displayA)
panelA.pack()
panelB = Label(displayB)
panelB.pack()

# create controls
openImgBut = Button(displayC, text="Open Image", padx=10, pady=5, command=selectImage)
openImgBut.pack(side="top", pady=5)

operateImgBut = Button(displayC, text="Operate Image", padx=10, pady=5, command=operateImg)
operateImgBut.pack(side="top", pady=5)

opImgSelect = OptionMenu(displayC, currImgOp, "CannyEdge Detection", "Red Mask")
opImgSelect.pack(side="top", pady=5)

divider = Label(displayC, text="~~~~~~~~~~~~~~~~~~~~~~~~")
divider.pack(side="top", pady=10)

openVidBut = Button(displayC, text="Open Video", padx=10, pady=5, command=selectVideo)
openVidBut.pack(side="top", pady=5)

operateVidBut = Button(displayC, text="Operate Video", padx=10, pady=5, command=operateVid)
operateVidBut.pack(side="top", pady=5)

opVidSelect = OptionMenu(displayC, currVidOp, "CannyEdge Detection", "BackgroundSub MOG", "BackgroundSub MOG2", "Draw Ball Outline", "Track Ball Flight")
opVidSelect.pack(side="top", pady=5)

pauseBut = Button(displayC, text="||", padx=10, pady=5, command=playPause)
pauseBut.pack(side="top", pady=5)

divider2 = Label(displayC, text="~~~~~~~~~~~~~~~~~~~~~~~~")
divider2.pack(side="top", pady=10)

showInputBut = Button(displayC, text="Hide Input Display", padx=10, pady=5, command=showDisplayA)
showInputBut.pack(side="top", pady=5)

showOutputBut = Button(displayC, text="Hide Output Display", padx=10, pady=5, command=showDisplayB)
showOutputBut.pack(side="top", pady=5)


root.mainloop()