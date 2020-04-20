# ------------------------------------------------------
#  Squash Vision v1.10.0
#
#  2020 Luke Gibson, Southampton, England
#  lg1n17@soton.ac.uk
# ------------------------------------------------------

from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tkinter as tk
import math

import v11_preProcessing as pre
import v11_postProcessing as post


def getResizeDim(image):
    '''
    Calculates the dimension required to scale an image such that it fits into the current display size.

    :param image: the image to be scaled
    :returns: the scale as a pair of ints
    '''
    global showInput

    # calculate scales required to fit image in current frame
    frameWidth = displayB.winfo_width()
    frameHeight = displayB.winfo_height()
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
    '''
    Button function to toggle showing the input video display.
    Updates input and output display sizes and locations to fit either the one or two displays.
    '''
    global showInput, displayA, displayB, panelA, panelB

    showInput = not showInput

    if showInput:
        showInputBut.configure(text="Disable Input Display")
    else:
        showInputBut.configure(text="Enable Input Display")
    
    # destroy existing display frames and their panels
    if displayA is not None:
        displayA.destroy()
        panelA.destroy()

    displayB.destroy()
    panelB.destroy()

    # reset frames and thier panels to new size and location
    displayA = tk.Frame(root, bg="#cccccc")
    displayB = tk.Frame(root, bg="#cccccc")

    if showInput:
        displayA.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.01)
        displayB.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.51)
    else:
        displayB.place(relwidth=0.75, relheight=0.98, relx=0.01, rely=0.01)

    panelA = tk.Label(displayA)
    panelA.pack()
    panelB = tk.Label(displayB)
    panelB.pack()


def showImage(cv2Image):
    '''Resize and convert an Open CV image to be displayed in the current display frame(s).'''
    # retrive the dimension which to resize the image for current display frame dimension
    resizeDim = getResizeDim(cv2Image)
    image = cv2.resize(cv2Image, resizeDim, interpolation=cv2.INTER_AREA)

    # convert to Tkinter format
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    panelA.configure(image=image)
    panelA.image = image
    panelB.configure(image=image)
    panelB.image = image
    

def selectVideo():
    '''Set the current video file filepath using a file dialog.'''
    global currVidFile

    # open a file selection box
    filename = filedialog.askopenfilename(title="Select Video")
    currVidFile = filename

    # check file was selected
    if len(currVidFile) > 0:
        print("Selected: ", currVidFile)
        capture = cv2.VideoCapture(currVidFile)

        # check capture opened and load first frame into display frames
        if not capture.isOpened():
            print("Can't open video: ", currVidFile)
        else:
            ret, frame = capture.read()

            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                showImage(image)

            capture.release()


def playPause():
    '''Button function that toggles pause.'''
    global pause

    pause = not pause
    if pause:
        nextFrameBut.configure(state=tk.NORMAL)
    else:
        nextFrameBut.configure(state=tk.DISABLED)


def showNextFrame():
    '''Button function that shows the next frame of the video.'''
    global nextFrame

    nextFrame = True


def operateVid():
    '''
    Button function that starts the video anaslysis process initalising all video data stores.
    Disables non-playback controll buttons.
    '''
    global currVidFile, nextFrame, pause, stage

    # reset video playback variables
    currCapture = cv2.VideoCapture(currVidFile)
    nextFrame = False
    pause = False

    # initalise video data stores
    bgSubMOG = cv2.bgsegm.createBackgroundSubtractorMOG()
    lastSeenBall = (-1,-1)
    trackPoints = []
    trackPredPoints = []
    linePoints = []
    deltaPoints = []
    gradPoints = []
    rateGradPoints = []
    contactFrames = []
    contactPrints = []
    outProb = 0
    frameIndex = 0
    
    
    if not currCapture.isOpened():
        print("Can't open video: ", currVidFile)
    else:
        # disable/enable running buttons
        openVidBut.configure(state=tk.DISABLED)
        operateVidBut.configure(state=tk.DISABLED)
        showInputBut.configure(state=tk.DISABLED)
        pauseBut.configure(state=tk.NORMAL)
        nextFrameBut.configure(state=tk.DISABLED)

        playVideo(currCapture, bgSubMOG, lastSeenBall, trackPoints, trackPredPoints, linePoints, gradPoints, rateGradPoints, deltaPoints, contactFrames, contactPrints, outProb, frameIndex)


def playVideo(currCapture, bgSubMOG, lastSeenBall, trackPoints, trackPredPoints, linePoints, gradPoints, rateGradPoints, deltaPoints, contactFrames, contactPrints, outProb, frameIndex):
    '''
    Loops through a video's frames twice. First loop collects the raw data populating the videos data stores.
    Once first loop finishes processes raw data to predict contact data, then initiates second loop.
    Second loop plays back the desired output image and calculates the probabilty the ball in the video.

    :param currCapture: the current open cv capture object
    :param bgSubMOG: the current open cv background subtractor object
    :param lastSeenBall: the (x,y) coordinates of the centre of the ball in the last frame
    :param trackPoints: the list of each frames detected ball center and radius
    :param trackPredPoints: the list of each frames predicted ball center and radius
    :param linePoints: the list of each frames line contour object
    :param gradPoints: the list of the gradient of the balls flight in each frame
    :param rateGradPoints: the list of the rate of change of gradient of the balls flight in each frame
    :param deltaPoints: the list of the stright line distance between current and last frames ball centre
    :param contactFrames: the list of pairs of (frame number, radius percentage of contact) where the ball contacts the wall
    :param contactPrints: the list of ball imprint masks
    :param outProb: the probability the ball was out 
    :param frameIndex: the current frame number of playback
    '''
    global pause, nextFrame, stage, currVidOp

    # stop playback if paused
    if not pause or nextFrame:
        nextFrame = False
        ret, frame = currCapture.read()

        # check to see if capture has ended
        if not ret:
            currCapture.release()
            frameIndex = 0
            
            # perform preprocessing steps on collected data
            if stage == 0:
                trackPoints, linePoints = pre.expandTrackGaps(trackPoints, linePoints)
                trackPredPoints = pre.fillTrackGaps(trackPoints)

                gradPoints = pre.calcPointGrad(trackPredPoints)
                gradPoints = pre.removeListNoise(gradPoints)
                rateGradPoints = pre.calcPointRateGrad(gradPoints)
                deltaPoints = pre.calcDeltaPoints(trackPredPoints)
                contactFrames = pre.calcContactFrames(rateGradPoints, deltaPoints)

                # update stage for post processing and restart playback
                stage = 1
                currCapture = cv2.VideoCapture(currVidFile)
                playVideo(currCapture, bgSubMOG, lastSeenBall, trackPoints, trackPredPoints, linePoints, gradPoints, rateGradPoints, deltaPoints, contactFrames, contactPrints, outProb, frameIndex)
            
            # finish playback and enable disabled controlls
            elif stage == 1:
                openVidBut.configure(state=tk.NORMAL)
                operateVidBut.configure(state=tk.NORMAL)
                showInputBut.configure(state=tk.NORMAL)
                pauseBut.configure(state=tk.DISABLED)
                nextFrameBut.configure(state=tk.DISABLED)
                stage = 0

        else:
            # get the operated image and updated video data
            if stage == 0:
                operatedImage, bgSubMOG, trackPoints, lastSeenBall, linePoints = pre.collectData(
                    frame, bgSubMOG, trackPoints, lastSeenBall, linePoints)
            if stage == 1:
                vidOp = currVidOp.get()
                operatedImage, trackPoints, trackPredPoints, linePoints, gradPoints, rateGradPoints, deltaPoints, frameIndex, contactFrames, contactPrints, outProb = post.showResult(
                    frame, trackPoints, trackPredPoints, linePoints, gradPoints, rateGradPoints, deltaPoints, frameIndex, contactFrames, contactPrints, outProb, vidOp)
            

            # resize + convert the operated image for display
            resizeDim = getResizeDim(frame)
            image = cv2.resize(frame, resizeDim, interpolation=cv2.INTER_AREA)
            operatedImage = cv2.resize(operatedImage, resizeDim, interpolation=cv2.INTER_AREA)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image=image)
            operatedImage = Image.fromarray(operatedImage)
            operatedImage = ImageTk.PhotoImage(image=operatedImage)

            # display original and operated images
            panelA.image = image
            panelA.configure(image=image)
            panelB.image = operatedImage
            panelB.configure(image=operatedImage)

            # after 5ms recall this function to playthrough next video frame
            root.after(5, playVideo, currCapture, bgSubMOG, lastSeenBall, trackPoints, trackPredPoints, linePoints, gradPoints, rateGradPoints, deltaPoints, contactFrames, contactPrints, outProb, frameIndex)
    else:
        root.after(500, playVideo, currCapture, bgSubMOG, lastSeenBall, trackPoints, trackPredPoints, linePoints, gradPoints, rateGradPoints, deltaPoints, contactFrames, contactPrints, outProb, frameIndex)



# initialise tkinter app
root = tk.Tk()
root.title("Squash Vision")

# declare global variables for playback controll
currVidFile = ""
currVidOp = tk.StringVar()
currVidOp.set("Make Decision")
pause = False
nextFrame = False
showInput = True
stage = 0

# calculate relative app dimensions for screen resolution
screenHeight = root.winfo_screenheight()
screenWidth = root.winfo_screenwidth()

rHeight = round(screenHeight/1.2)
rWidth = round(screenWidth/1.8)
rDim = str(rWidth) + "x" + str(rHeight)

root.geometry(rDim)

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

# create user controls
openVidBut = tk.Button(displayC, text="Select Shot", padx=10, pady=5, command=selectVideo)
openVidBut.pack(side="top", pady=5)

operateVidBut = tk.Button(displayC, text="START", padx=10, pady=5, command=operateVid)
operateVidBut.pack(side="top", pady=5)

divider1 = tk.Label(displayC, text="-----  Playback Controll's  -----")
divider1.pack(side="top", pady=10)

pauseBut = tk.Button(displayC, text="Play/Pause", padx=10, pady=5, command=playPause, state=tk.DISABLED)
pauseBut.pack(side="top", pady=5)

nextFrameBut = tk.Button(displayC, text="Next Frame", padx=10, pady=5, command=showNextFrame, state=tk.DISABLED)
nextFrameBut.pack(side="top", pady=5)

divider2 = tk.Label(displayC, text="-----  Display Controll's  -----")
divider2.pack(side="top", pady=10)

showInputBut = tk.Button(displayC, text="Disable Input Display", padx=10, pady=5, command=showDisplayA)
showInputBut.pack(side="top", pady=5)

opVidSelect = tk.OptionMenu(displayC, currVidOp, "Make Decision", "Probability Mask", "Ball Trajectory", "Object Detection")
opVidSelect.pack(side="top", pady=5)

root.mainloop()