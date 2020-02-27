from tkinter import filedialog
from PIL import Image, ImageTk
from sys import exit
import cv2
import numpy as np
import tkinter as tk
import time


# Display Helper Functions


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


def operateVid():
    global currVidFile, currCapture, bgSubMOG, currVidOp, trackList, predPoints, predPointsIndex, trackPoints

    # Release capture and create new one capture
    currCapture = cv2.VideoCapture(currVidFile)

    # Reset tracks
    trackList = []

    trackPoints = []
    predPoints = []
    predPointsIndex = 0
    bgSubMOG = cv2.bgsegm.createBackgroundSubtractorMOG()

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
    global panelA, panelB, root, currCapture, currVidOp, pause, predPoints, predPointsIndex, trackPoints

    if not pause:
        ret, frame = currCapture.read()

        if not ret:
            currCapture.release()
            print("Capture Ends")

            # If finishing generate track then draw the new track
            if currVidOp.get() == 'Draw Ball Prediction':
                # update predPoints now trackPoints are populated
                print("Predicting Points")
                predPoints = fillTrackGaps(trackPoints)

                # reset capture and call playVideo now as display predPoints options
                print("Drawing Ball Full")
                currVidOp.set('Draw Ball Prediction 2')
                currCapture = cv2.VideoCapture(currVidFile)
                playVideo()
            # reset currOperation to stage 1 of "Draw Ball Predicition"
            elif currVidOp.get() == 'Draw Ball Prediction 2':
                currVidOp.set('Draw Ball Prediction')
            
        else:
            resizeDim = getResizeDim(frame)
            operatedImage = frame

            # perform operation
            if currVidOp.get() == 'Draw Ball Outline':
                operatedImage = drawBallVid(frame)
            elif currVidOp.get() == 'Draw Line Outline':
                operatedImage = lineDetectVid(frame)
            elif currVidOp.get() == 'Track Ball Flight':
                operatedImage = trackBallVid(frame)
            elif currVidOp.get() == 'Draw Ball Prediction':
                operatedImage = genTrackVid(frame)
            elif currVidOp.get() == 'Draw Ball Prediction 2':
                operatedImage = drawBallFullVid(frame)

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


def genTrackVid(frame):
    global currCapture, bgSubMOG, trackPoints

    # blur and convert to grayscale
    frameBlurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frameGray = cv2.cvtColor(frameBlurred, cv2.COLOR_BGR2GRAY)

    # Create binary mask using subtractor
    mask = bgSubMOG.apply(frameGray)

    # Perform morphological opening (erosion followed by dilation) - to remove noise from mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # generate output image
    height, width = frame.shape[:2]
    outputFrame = np.zeros((height,width), np.uint8)

    # find contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # if mask is not empty then find the ball center and radius and add this to the track - else add the 
    if len(contours) > 0:
        largestCon = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largestCon)
        cv2.circle(outputFrame, (int(x),int(y)), int(radius), 255, -1)
        trackPoints.append((int(x), int(y), int(radius)))
    else:
        trackPoints.append((-1,-1,0))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(outputFrame, "Generating Track", (20,70), font, 2, 255, 2, cv2.LINE_AA)

    return outputFrame


def fillTrackGaps(trackPoints):
    cleanTrackPoints = []
    cleanTrackPoints.append(trackPoints[0])

    # if either neighbouring point doesn't have a ball detection assume the detection in current point is poor
    for i in range(1, len(trackPoints) - 1):
        x, y, r = trackPoints[i]

        if trackPoints[i-1] != (-1,-1,0) and trackPoints[i+1] != (-1,-1,0):
            cleanTrackPoints.append((x,y,r))
        else:
            cleanTrackPoints.append((-1,-1,0))
    
    cleanTrackPoints.append(trackPoints[-1])

    # find all sections with a missing point
    missingSections = [[]]
    sectionCount = 0

    for i in range(len(cleanTrackPoints) - 1):
        x, y, r = cleanTrackPoints[i]
        pX, pY, pR = cleanTrackPoints[i - 1]
        nX, nY, nR = cleanTrackPoints[i + 1]

        # adds missing points to section
        if x < 0:
            missingSections[sectionCount].append((x,y,r,i))
        # adds real value far to end of section and increments section count
        elif pX < 0:
            missingSections[sectionCount].append((x,y,r,i))
            sectionCount += 1
        # adds real value at start of section
        elif nX < 0:
            missingSections.append([])
            missingSections[sectionCount].append((x,y,r,i))
    
    # predict values for points in the missing sections
    for section in missingSections:
        # excludes the first points where ball hasn't been found yet
        if section[0][0] != -1 and section[-1][0] != -1:

            startX, startY, startR, startPos = section[0]
            endX, endY, endR, endPos = section[-1]
            numMissing = len(section) - 2

            xStep = (endX - startX) / (numMissing + 1)
            yStep = (endY - startY) / (numMissing + 1)
            
            # calculate ball position at even spacing for missing values (assumes a stright line)
            for i in range(1, len(section) - 1):
                pos = section[i][3]
                missingX = int(startX + (i * xStep))
                missingY = int(startY + (i * yStep))
                # assign predicted radius based on which end of the gap current point is closest too
                if i*2 <= numMissing:
                    section[i] = (missingX, missingY, startR)
                else:
                    section[i] = (missingX, missingY, endR)
                
                # rewrite missing point in full list
                cleanTrackPoints[pos] = section[i]
                      
    return cleanTrackPoints


# uses the pre calculated predicted ball points
def drawBallFullVid(frame):
    global predPoints, predPointsIndex

    outputFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x, y, r = predPoints[predPointsIndex]

    if x > -1:
        cv2.circle(outputFrame, (x,y), r, (0, 255, 0), 1)

    predPointsIndex += 1

    return outputFrame
        

def drawBallVid(frame):
    global bgSubMOG

    # blur and convert to grayscale
    frameBlurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frameGray = cv2.cvtColor(frameBlurred, cv2.COLOR_BGR2GRAY)

    # Create binary mask using subtractor
    mask = bgSubMOG.apply(frameGray)

    # Perform morphological opening (erosion followed by dilation) - to remove noise from mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # find contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # generate output frame
    outputFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if len(contours) > 0:
        largestCon = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largestCon)

        M = cv2.moments(largestCon)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        cv2.circle(outputFrame, (int(x), int(y)), int(radius), (0, 255, 0), 1)
        cv2.circle(outputFrame, center, 1, (255, 0, 0), -1)

    return outputFrame


def trackBallVid(frame):
    global bgSubMOG, trackList

    # blur and convert to grayscale
    frameBlurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frameGray = cv2.cvtColor(frameBlurred, cv2.COLOR_BGR2GRAY)

    # Create binary mask using subtractor
    mask = bgSubMOG.apply(frameGray)

    # Perform morphological opening (erosion followed by dilation) - to remove noise from mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # find contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # generate output frame
    outputFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if len(contours) > 0:
        largestCon = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largestCon)

        cv2.circle(outputFrame, (int(x), int(y)), int(radius), (0, 255, 0), 1)
        trackList.append((int(x), int(y)))

    for point1, point2 in zip(trackList, trackList[1:]): 
        cv2.line(outputFrame, point1, point2, (0, 0, 255), 2) 

    return outputFrame


def lineDetectVid(frame):
    # create output frame
    outputFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
            cv2.drawContours(outputFrame, [c], -1, (0, 255, 0), 1)
    
    return outputFrame


# Initialise App
root = tk.Tk()

# Global Variables
currVidFile = ""
currVidOp = tk.StringVar()
currVidOp.set("Draw Ball Prediction")
currCapture = cv2.VideoCapture(0)
pause = False

bgSubMOG = cv2.bgsegm.createBackgroundSubtractorMOG()
trackList = []
trackPoints = []
predPoints = []
predPointsIndex = 0
waitBool = True

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
openVidBut = tk.Button(displayC, text="Open Video", padx=10, pady=5, command=selectVideo)
openVidBut.pack(side="top", pady=5)

operateVidBut = tk.Button(displayC, text="Operate Video", padx=10, pady=5, command=operateVid)
operateVidBut.pack(side="top", pady=5)

opVidSelect = tk.OptionMenu(displayC, currVidOp, "Draw Ball Outline", "Draw Line Outline", "Track Ball Flight", 'Draw Ball Prediction')
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