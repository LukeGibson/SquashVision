from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np


# Shared Operations


def getResizeDim(image):
    # get frame dimension
    frameWidth = displayA.winfo_width()
    frameHeight = displayA.winfo_height()

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


def operateImg():
    global currImgOp

    if currImgOp.get() == "CannyEdge Detection":
        cannyEdgeImg()


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


def playPause():
    global pause
    pause = not pause
    print("Video paused: ", pause)


def playVideo():
    global panelA, panelB, root, currCapture, currVidOp, pause, bgSubMOG, bgSubMOG2

    if not pause:
        ret, frame = currCapture.read()

        if not ret:
            print("Video End")
        else:
            resizeDim = getResizeDim(frame)
            operatedImage = frame

            # perform operation
            if currVidOp.get() == 'CannyEdge Detection':
                operatedImage = cannyEdgeVid(frame, resizeDim)
            elif currVidOp.get() == 'BackgroundSub MOG':
                operatedImage = backgroundSubVid(frame, resizeDim, bgSubMOG)
            elif currVidOp.get() == 'BackgroundSub MOG2':
                operatedImage = backgroundSubVid(frame, resizeDim, bgSubMOG2)

            # resize
            image = cv2.resize(frame, resizeDim, interpolation=cv2.INTER_AREA)

            # convert format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image=image)

            # display original frame
            panelA.image = image
            panelA.configure(image=image)

            # display operated frame
            panelB.image = operatedImage
            panelB.configure(image=operatedImage)

            root.after(5, playVideo)
    else:
        root.after(500, playVideo)


def cannyEdgeVid(frame, resizeDim):
    # perform edge detection
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameEdges = cv2.Canny(frameGray, 200, 250)

    # resize
    operatedImage = cv2.resize(frameEdges, resizeDim, interpolation=cv2.INTER_AREA)

    # Convert format
    operatedImage = Image.fromarray(operatedImage)
    operatedImage = ImageTk.PhotoImage(operatedImage)

    return operatedImage


def backgroundSubVid(frame, resizeDim, subtractor):
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create binary mask using subtractor
    mask = subtractor.apply(frameGray)

    # Perform morphological opening (erosion followed by dilation) - to remove noise from mask
    kernel = np.ones((5, 5), np.uint8)
    maskOpening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # resize
    operatedImage = cv2.resize(maskOpening, resizeDim, interpolation=cv2.INTER_AREA)

    # Convert format
    operatedImage = Image.fromarray(operatedImage)
    operatedImage = ImageTk.PhotoImage(operatedImage)

    return operatedImage


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
displayA.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.01)

displayB = Frame(root, bg="#cccccc")
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

opImgSelect = OptionMenu(displayC, currImgOp, "CannyEdge Detection")
opImgSelect.pack(side="top", pady=5)

divider = Label(displayC, text="~~~~~~~~~~~~~~~~~~~~~~~~")
divider.pack(side="top", pady=10)

openVidBut = Button(displayC, text="Open Video", padx=10, pady=5, command=selectVideo)
openVidBut.pack(side="top", pady=5)

operateVidBut = Button(displayC, text="Operate Video", padx=10, pady=5, command=playVideo)
operateVidBut.pack(side="top", pady=5)

opVidSelect = OptionMenu(displayC, currVidOp, "CannyEdge Detection", "BackgroundSub MOG", "BackgroundSub MOG2")
opVidSelect.pack(side="top", pady=5)

playBut = Button(displayC, text="I> / ||", padx=10, pady=5, command=playPause)
playBut.pack(side="top", pady=5)

root.mainloop()
