from tkinter import filedialog
from PIL import Image, ImageTk
from sys import exit
import cv2
import numpy as np
import tkinter as tk
import time
import math
import numba as nb

import v9_calculations as calc


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


def playPause():
    global pause, pauseBut

    pause = not pause
    if pause:
        print("Play")
        pauseBut.configure(text="I>")
    else:
        print("Pause")
        pauseBut.configure(text="||")


def showNextFrame():
    global nextFrame

    print("Next Frame")
    nextFrame = True

def stopOperation():
    global stop

    print("Stop")
    stop = True


def operateVid():
    global currVidFile, currCapture, bgSubMOG, currVidOp, stop, nextFrame, lastSeenBall, predPoints, trackPoints, linePoints, anglePoints, rateAnglePoints, gradPoints, rateGradPoints, deltaPoints, frameIndex, contactFrames, contactPrints, outProb

    # set stop to true
    stop = False
    nextFrame = False

    # Release capture and create new one capture
    currCapture = cv2.VideoCapture(currVidFile)
    bgSubMOG = cv2.bgsegm.createBackgroundSubtractorMOG()

    # holds the recorded ball centeroids in each frame
    trackPoints = []
    # holds the predicted ball centeroids in each frame
    predPoints = []
    # store an array of points detected as the line for each frame
    linePoints = []
    # stores the number of pixles moved between frames
    deltaPoints = []
    # stores the angle of local ball travel between frames
    anglePoints = []

    rateAnglePoints = []
    gradPoints = []
    rateGradPoints = []

    # holds the frame indices of the frames where 'contact' is detected
    contactFrames = []

    # reset the last seen ball coordinate
    lastSeenBall = (-1,-1)

    # reset contactPrints
    contactPrints = []
    outProb = 0

    # stores the current frame index - to reference list values
    frameIndex = 0

    # Check capture opened and load display first frame
    if not currCapture.isOpened():
        print("Can't open video: ", currVidFile)
    else:
        playVideo()


def playVideo():
    global panelA, panelB, root, currCapture, currVidOp, pause, nextFrame, stop, predPoints, trackPoints, linePoints, anglePoints, rateAnglePoints, gradPoints, rateGradPoints, deltaPoints, contactFrames, frameIndex

    if not stop:
        if not pause or nextFrame:
            nextFrame = False
            ret, frame = currCapture.read()

            if not ret:
                currCapture.release()
                frameIndex = 0
                print("Capture Ends")
                
                # Do next stage of the operation process
                if currVidOp.get() == 'Make Decision':
                    # expand gaps in track points and overwrite distorted linePoints
                    trackPoints, linePoints = calc.expandTrackGaps(trackPoints, linePoints)

                    # predicit missing points in track
                    predPoints = calc.fillTrackGaps(trackPoints)

                    # calculate ball data lists
                    gradPoints = calc.calcPointGrad(predPoints)
                    gradPoints = calc.removeListNoise(gradPoints)
                    rateGradPoints = calc.calcPointRateGrad(gradPoints)
                    deltaPoints = calc.calcDeltaPoints(predPoints)

                    # get frames of contact and compression in frame
                    contactFrames = calc.calcContactFrames(rateGradPoints, deltaPoints)
                    print("Contact (Frames, Radius %):", contactFrames)

                    # use predPoints, linePoints and contactFrames to calculate decision in each frame
                    currVidOp.set('Make Decision 2')
                    currCapture = cv2.VideoCapture(currVidFile)
                    playVideo()
                
                elif currVidOp.get() == 'Make Decision 2':
                    currVidOp.set('Make Decision')

            else:
                resizeDim = getResizeDim(frame)
                operatedImage = frame

                # perform operation
                if currVidOp.get() == 'Make Decision':
                    operatedImage = generateTrackVid(frame)
                if currVidOp.get() == 'Make Decision 2':
                    operatedImage = decisionVid(frame)
                

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
    else:
        currCapture.release()
        frameIndex = 0


def decisionVid(frame):
    global predPoints, linePoints, frameIndex, contactFrames, contactPrints, outProb

    # get frame dimensions
    height, width = frame.shape[:2]

    # get frame line and ball data
    ballData = predPoints[frameIndex]
    lineData = linePoints[frameIndex]


    # if frame is in contact add print to to contactPrints
    if frameIndex in [i[0] for i in contactFrames]:
        ballContact = True
        # get contactFrame data
        radiusPercent = contactFrames[[i[0] for i in contactFrames].index(frameIndex)][1]

        # create then contactPrint mask
        ballCenter = ballData[:2]
        ballRadius = ballData[2]
        ballPrintRadius = int(math.ceil(ballRadius * (radiusPercent / 100)))

        ballPrintMask = np.zeros((height, width), np.uint8)
        cv2.circle(ballPrintMask, ballCenter, ballPrintRadius, 255, -1)

        ballPrintMaskCol = np.zeros((height, width, 3), np.uint8)
        cv2.circle(ballPrintMaskCol, ballCenter, ballPrintRadius, (0,0,255), -1)

        # add current ballPrintMask to list to be used for remaining contactFrames
        contactPrints.append((ballPrintMask, ballPrintMaskCol))
    else:
        ballContact = False
    

    # sum accumulated print masks to create the contactMask
    contactMask = np.zeros((height, width), np.uint8)
    contactMaskCol = np.zeros((height, width, 3), np.uint8)

    for mask, colMask in contactPrints:
        contactMask = cv2.add(mask, contactMask)
        contactMaskCol = cv2.add(colMask, contactMaskCol)
    
    # join up indevidual prints into a single print
    kernel = np.ones((7,7), np.uint8)
    contactMask = cv2.dilate(contactMask, kernel, iterations=2)
    contactMask = cv2.erode(contactMask, kernel, iterations=2)
    contactMaskCol = cv2.dilate(contactMaskCol, kernel, iterations=2)
    contactMaskCol = cv2.erode(contactMaskCol, kernel, iterations=2)

    # create line mask
    lineMask = np.zeros((height, width), np.uint8)
    cv2.drawContours(lineMask, [lineData], -1, 255, 1)


    # Find if ball is out
    if frameIndex in [i[0] for i in contactFrames]:
        # calculate the probability the ball was out
        probMask, newOutProb = calc.probOut(contactMask, lineMask)

        # update outProb if the new frame probability is larger
        if newOutProb > outProb:
            outProb = newOutProb
    
    
    # create colored masks for output
    lineMaskCol = np.zeros((height, width, 3), np.uint8)
    cv2.drawContours(lineMaskCol, [lineData], -1, (255,0,0), 1)

    ballMaskCol = np.zeros((height, width, 3), np.uint8)
    cv2.circle(ballMaskCol, ballData[:2], ballData[2], (0,255,0), 1)

    # add colered masks together
    output = cv2.add(ballMaskCol, lineMaskCol)
    output = cv2.add(output, contactMaskCol, 1)

    # write text on output
    font = cv2.FONT_HERSHEY_SIMPLEX

    if ballContact:
        cv2.putText(output, "Contact: True", (20,70), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(output, "Contact: False", (20,70), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        
    if outProb >= 0.5:
        cv2.putText(output, "Prob OUT: " + str(round(outProb * 100, 0)) +"%", (20,140), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(output, "Prob OUT: " + str(outProb * 100) +"%", (20,140), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(output, "Frame: " + str(frameIndex), (20,210), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    frameIndex += 1
    
    return output


def generateTrackVid(frame):
    global bgSubMOG, trackPoints, lastSeenBall, linePoints

    # generate output image
    height, width = frame.shape[:2]
    outputFrame = np.zeros((height,width), np.uint8)


    ## Generate line points


    # threshold on red color
    lowColor = (0,0,75)
    highColor = (50,50,135)
    mask = cv2.inRange(frame, lowColor, highColor)

    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # to join line contours objects
    mask = cv2.dilate(mask, kernel,iterations=4)
    mask = cv2.erode(mask, kernel, iterations=4)

    # get contours
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 2:
        contours = contours[0]
    else:
        contours = contours[1]
    
    largestSpan = 0
    largestSpanCon = None
    
    # find contour with largest horizontail span - a feature of the outliney
    for c in contours:
        leftmost = (c[c[:,:,0].argmin()][0])[0]
        rightmost = (c[c[:,:,0].argmax()][0])[0]
        span = abs(leftmost - rightmost)

        if span > largestSpan:
            largestSpan = span
            largestSpanCon = c
    
    # draw contour with largest span
    if len(contours) > 0:
        cv2.drawContours(outputFrame, [largestSpanCon], -1, 128, -1)

    linePoints.append(largestSpanCon)
    

    ## Generate track points


    # blur and convert to grayscale
    frameBlurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frameGray = cv2.cvtColor(frameBlurred, cv2.COLOR_BGR2GRAY)

    # Create binary mask using subtractor
    mask = bgSubMOG.apply(frameGray)

    # Perform morphological opening (erosion followed by dilation) - to remove noise from mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # to join non ball objects
    mask = cv2.dilate(mask, kernel,iterations=4)
    mask = cv2.erode(mask, kernel, iterations=4)

    # find contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:

        possibleBallCons = []
        for con in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(con)

            if radius > 3 and radius < 10:
                # add to list of possible balls
                possibleBallCons.append(con)

        if len(possibleBallCons) > 0:

            # store the contour closest to the last known ball
            found = False
            closestCon = None
            # theshold for minimum distance from last known ball - could be adapted to increase when number of frames since last detected
            smallestDelta = 200
            nextBall = (-1,-1)

            lastX, lastY = lastSeenBall
            
            # calculate the center for each possible ball
            for con in possibleBallCons:
                M = cv2.moments(con)
                x = int(M["m10"] / M["m00"]) 
                y = int(M["m01"] / M["m00"])

                delta = math.sqrt(((x - lastX)**2) + ((y - lastY)**2))

                # keep track of closest ball to last known ball
                if delta < smallestDelta or lastSeenBall == (-1,-1):
                    smallestDelta = delta
                    closestCon = con
                    nextBall = (x,y)
                    found = True
            
            if found:
                # update the global last seen ball
                lastSeenBall = nextBall

                # draw ball cloest possbile contour (if found within a threshold)
                ((x, y), radius) = cv2.minEnclosingCircle(closestCon)

                # M = cv2.moments(closestCon)
                # center = ((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
                cv2.circle(outputFrame, (int(x), int(y)), int(radius), 255, -1)

                trackPoints.append((int(x), int(y), int(radius)))
            else:
                trackPoints.append((-1,-1,0))
        else:
            trackPoints.append((-1,-1,0))
    else:
        trackPoints.append((-1,-1,0))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(outputFrame, "Generating Track", (20,70), font, 2, 255, 2, cv2.LINE_AA)

    return outputFrame


# Initialise App
root = tk.Tk()

# Global Variables
currVidFile = ""
currVidOp = tk.StringVar()
currVidOp.set("Make Decision")
currCapture = cv2.VideoCapture(0)
pause = False
nextFrame = False
stop = False

bgSubMOG = cv2.bgsegm.createBackgroundSubtractorMOG()

# stores the calculated data about frames of the input video
trackPoints = []
predPoints = []
linePoints = []
anglePoints = []
deltaPoints = []

rateAnglePoints = []
gradPoints = []
rateGradPoints = []

# holds the frame indices of the frames where 'contact' is detected
contactFrames = []
frameIndex = 0

# last known ball centre
lastSeenBall = (-1,-1)

# stores the accumulating contact prints of ball
contactPrints = []
outProb = 0

showA = True
showB = True

# calculate relative app dimensions for screen resolution
screenHeight = root.winfo_screenheight()
screenWidth = root.winfo_screenwidth()

rHeight = round(screenHeight/1.2)
rWidth = round(screenWidth/1.8)
rDim = str(rWidth) + "x" + str(rHeight)

root.geometry(rDim)
root.title("Squash Vision")

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
openVidBut = tk.Button(displayC, text="Open Video", padx=10, pady=5, command=selectVideo)
openVidBut.pack(side="top", pady=5)

operateVidBut = tk.Button(displayC, text="Operate Video", padx=10, pady=5, command=operateVid)
operateVidBut.pack(side="top", pady=5)

opVidSelect = tk.OptionMenu(displayC, currVidOp, "Make Decision")
opVidSelect.pack(side="top", pady=5)

pauseBut = tk.Button(displayC, text="||", padx=10, pady=5, command=playPause)
pauseBut.pack(side="top", pady=5)

nextFrameBut = tk.Button(displayC, text=">>", padx=10, pady=5, command=showNextFrame)
nextFrameBut.pack(side="top", pady=5)

stopBut = tk.Button(displayC, text="STOP", padx=10, pady=5, command=stopOperation)
stopBut.pack(side="top", pady=5)

divider2 = tk.Label(displayC, text="~~~~~~~~~~~~~~~~~~~~~~~~")
divider2.pack(side="top", pady=10)

showInputBut = tk.Button(displayC, text="Hide Input Display", padx=10, pady=5, command=showDisplayA)
showInputBut.pack(side="top", pady=5)

showOutputBut = tk.Button(displayC, text="Hide Output Display", padx=10, pady=5, command=showDisplayB)
showOutputBut.pack(side="top", pady=5)


root.mainloop()