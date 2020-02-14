from tkinter import filedialog
from PIL import Image, ImageTk
from sys import exit
import cv2
import numpy as np
import tkinter as tk


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

    displayA = tk.Frame(root, bg="#cccccc")
    displayB = tk.Frame(root, bg="#cccccc")

    if showA and showB:
        displayA.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.01)
        displayB.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.51)
    elif showA:
        displayA.place(relwidth=0.75, relheight=0.98, relx=0.01, rely=0.01)
    elif showB:
        displayB.place(relwidth=0.75, relheight=0.98, relx=0.01, rely=0.01)

    panelA = tk.Label(displayA)
    panelA.pack()
    panelB = tk.Label(displayB)
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 200, 250)

        showImage(edges, False)


def lineDetectImg():
    global currImgFile

    if len(currImgFile) > 0:
        # load image
        image = cv2.imread(currImgFile)
        output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # threshold on red color
        lowColor = (0,0,75)
        highColor = (50,50,135)
        mask = cv2.inRange(image, lowColor, highColor)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # get contours
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 2:
            contours = contours[0]
        else:
            contours = contours[1]
        
        # draw contour with largest area
        for c in contours:
            area = cv2.contourArea(c)
            if area > 5000:
                cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
        
        showImage(mask, False)


def operateImg():
    global currImgOp

    if currImgOp.get() == "CannyEdge Detection":
        cannyEdgeImg()
    if currImgOp.get() == "Line Detect":
        lineDetectImg()


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
            elif currVidOp.get() == 'Draw Line Outline':
                operatedImage = lineDetectVid(frame)
            elif currVidOp.get() == 'Track Ball Flight':
                operatedImage = trackBallVid(frame, bgSubMOG, trackList)
            elif currVidOp.get() == 'Ball In/Out':
                operatedImage = ballInOutVid(frame, bgSubMOG)
            


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


def lineDetectVid(frame):
    # create output frame
    output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    # threshold on red color
    lowColor = (0,0,85)
    highColor = (50,50,135)
    mask = cv2.inRange(frame, lowColor, highColor)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # get contours
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 2:
        contours = contours[0]
    else:
        contours = contours[1]
    
    # draw contour with largest area
    for c in contours:
        area = cv2.contourArea(c)
        if area > 5000:
            cv2.drawContours(output, [c], -1, (0, 255, 0), 1)
    
    return output


def getBallPoints(frame, subtractor):
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

    # create blank image
    height, width, channels = frame.shape
    blank = np.zeros((height, width),np.uint8)

    # draw circle on image
    if len(contours) > 0:
        largestCon = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largestCon)

        cv2.circle(blank, (int(x), int(y)), int(radius), 255, -1)
        
    # find all white points in frame
    ballPoints = np.transpose(np.where(blank==255))

    return (blank, ballPoints)


def getLinePoints(frame):
    # threshold on red color
    lowColor = (0,0,85)
    highColor = (50,50,135)
    mask = cv2.inRange(frame, lowColor, highColor)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # get contours
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create blank image
    height, width, channels = frame.shape
    blank = np.zeros((height, width), np.uint8)

    if len(contours) == 2:
        contours = contours[0]
    else:
        contours = contours[1]
    
    # draw contour with largest area
    for c in contours:
        area = cv2.contourArea(c)
        if area > 5000:
            cv2.drawContours(blank, [c], -1, 255, -1)
    
    # find all white points in frame
    linePoints = np.transpose(np.where(blank==255))
    
    return (blank, linePoints)


def getDecision(ballPoints, linePoints):
    for bPoint in ballPoints:
        by = bPoint[0]
        bx = bPoint[1]

        for lPoint in linePoints:
            ly = lPoint[0]
            lx = lPoint[1]

            if bx == lx and by <= ly:
                return "Ball Out"

    return "Ball In"


def ballInOutVid(frame, subtractor):
    # get the points and masks for ball and line
    ballMask, ballPoints = getBallPoints(frame, subtractor)
    lineMask, linePoints = getLinePoints(frame)

    # check all points to see if a ball violates in/out term
    decision = getDecision(ballPoints, linePoints)

    # combine masks for output image
    combinedMask = cv2.add(ballMask, lineMask)

    # write decision on frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combinedMask, decision, (20,70), font, 2, (255,255,255), 2, cv2.LINE_AA)
    
    return combinedMask


# Initialise App
root = tk.Tk()

# Global Variables
currImgFile = ""
currImgOp = tk.StringVar()
currImgOp.set("Line Detect")

currVidFile = ""
currVidOp = tk.StringVar()
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
displayA = tk.Frame(root, bg="#cccccc")
displayB = tk.Frame(root, bg="#cccccc")

displayA.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.01)
displayB.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.51)

displayC = tk.Frame(root)
displayC.place(relwidth=0.23, relheight=0.98, relx=0.76, rely=0.01)

# initialise labels to hold images
panelA = tk.Label(displayA)
panelA.pack()
panelB = tk.Label(displayB)
panelB.pack()

# create controls
openImgBut = tk.Button(displayC, text="Open Image", padx=10, pady=5, command=selectImage)
openImgBut.pack(side="top", pady=5)

operateImgBut = tk.Button(displayC, text="Operate Image", padx=10, pady=5, command=operateImg)
operateImgBut.pack(side="top", pady=5)

opImgSelect = tk.OptionMenu(displayC, currImgOp, "CannyEdge Detection", "Red Mask", "Line Detect")
opImgSelect.pack(side="top", pady=5)

divider = tk.Label(displayC, text="~~~~~~~~~~~~~~~~~~~~~~~~")
divider.pack(side="top", pady=10)

openVidBut = tk.Button(displayC, text="Open Video", padx=10, pady=5, command=selectVideo)
openVidBut.pack(side="top", pady=5)

operateVidBut = tk.Button(displayC, text="Operate Video", padx=10, pady=5, command=operateVid)
operateVidBut.pack(side="top", pady=5)

opVidSelect = tk.OptionMenu(displayC, currVidOp, "CannyEdge Detection", "BackgroundSub MOG", "BackgroundSub MOG2", "Draw Ball Outline", "Draw Line Outline", "Track Ball Flight", "Ball In/Out")
opVidSelect.pack(side="top", pady=5)

pauseBut = tk.Button(displayC, text="||", padx=10, pady=5, command=playPause)
pauseBut.pack(side="top", pady=5)

divider2 = tk.Label(displayC, text="~~~~~~~~~~~~~~~~~~~~~~~~")
divider2.pack(side="top", pady=10)

showInputBut = tk.Button(displayC, text="Hide Input Display", padx=10, pady=5, command=showDisplayA)
showInputBut.pack(side="top", pady=5)

showOutputBut = tk.Button(displayC, text="Hide Output Display", padx=10, pady=5, command=showDisplayB)
showOutputBut.pack(side="top", pady=5)


root.mainloop()