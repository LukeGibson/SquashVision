import numpy as np
import os
import math
import cv2
import v8_calculations as calc
import csv
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import randint


# returns the contact frames for a given path
def getContactFrames(clipPath):
    trackPoints = []
    predPoints = []
    linePoints = []
    deltaPoints = []
    anglePoints = []
    rateAnglePoints = []
    gradPoints = []
    rateGradPoints = []

    contactFrames = []
    lastSeenBall = (-1,-1)

    bgSubMOG = cv2.bgsegm.createBackgroundSubtractorMOG()
    currCapture = cv2.VideoCapture(clipPath)
    
    while True:
        ret, frame = currCapture.read()

        if ret:
            # FIND LINE POINTS

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

            linePoints.append(largestSpanCon)


            # FIND TRACK POINTS

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
                        trackPoints.append((int(x), int(y), int(radius)))
                    else:
                        trackPoints.append((-1,-1,0))
                else:
                    trackPoints.append((-1,-1,0))
            else:
                trackPoints.append((-1,-1,0))
        else:
            break
    
    currCapture.release()

    trackPoints, linePoints = calc.expandTrackGaps(trackPoints, linePoints)
    predPoints = calc.fillTrackGaps(trackPoints)

    deltaPoints = calc.calcDeltaPoints(predPoints)

    gradPoints = calc.calcPointGrad(predPoints)
    gradPoints = calc.removeListNoise(gradPoints)
    rateGradPoints = calc.calcPointRateGrad(gradPoints)
    # take compression distance 
    contactFrames, compression = calc.calcContactFrames(rateGradPoints, deltaPoints) # compression graph

    return (contactFrames, compression) # compression graph


def genCollectionResults(collectionName, testName):
    clipDirPath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\Footage\\' + collectionName + '\\'
    resultDirPath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\ContactTests\\' + testName + '\\csv\\'
    resultFilePath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\ContactTests\\' + testName + '\\csv\\' + collectionName + '.csv'


    # get every clip file name in clipDirPath
    clipFiles = []

    for r, d, f in os.walk(clipDirPath):
        for clip in f:
            clipFiles.append(str(clip))

    
    # stores the list of contact frames for each clip
    clipContactFrames = []
    clipCompressions = []  # compression graph

    for clip in clipFiles:
        print("Calculating Contact:", collectionName, clip)
        clipPath = clipDirPath + clip
        # take compression distance
        contactFrames, compression = getContactFrames(clipPath) # compression graph
        clipContactFrames.append(contactFrames)
        clipCompressions.append(compression)  # compression graph
        

    # generate csv rows
    rows = []

    for i in range(len(clipFiles)):
        if clipFiles[i][5] != '.':
            clipNum = int(clipFiles[i][4:6])
        else:
            clipNum = int(clipFiles[i][4])
        
        if len(clipContactFrames[i]) == 0:
            frameStart = 0
            frameEnd = 0
            compression = 0 # compression graph
        else:
            frameStart = clipContactFrames[i][0]
            frameEnd = clipContactFrames[i][-1]
            compDistance, radiusPercent = clipCompressions[i] # compression graph

        rows.append([clipNum, frameStart, frameEnd, compression])  # compression graph
    
    # sort into accending clip number order
    rows = sorted(rows, key = lambda x: x[0])
    
    # generate directory
    try:
        os.makedirs(resultDirPath)
    except OSError:
        print ("Dir Exists:", testName)
    else:
        print ("Dir Created:", testName)

    
    # write rows to a csv
    with open(resultFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


# compear a generated csv with the truth csv
def compearResult(testName, collections):
    outputFilePath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\ContactTests\\' + testName + '\\'  + testName + '_results.txt'
    allFrameDiffs = []
    allCompressions = []  # compression graph

    for collectionName in collections:
        resultFilePath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\ContactTests\\' + testName + '\\csv\\' + collectionName + '.csv'
        truthFilePath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\ContactTests\\truth\\' + 'truth_' + collectionName + '.csv'

        # read truth and test csv for configuration
        truthRows = []
        resultRows = []
        colCompressions = []

        for line in open(resultFilePath):
            row = line.split(',')
            clipNum = int(row[0])
            firstFrame = int(row[1])
            lastFrame = int(row[2])
            compression = float(row[3]) # compression graph
            
            colCompressions.append(compression)  # compression graph
            resultRows.append((clipNum, firstFrame, lastFrame)) 
        
        for line in open(truthFilePath):
            row = line.split(',')
            clipNum = int(row[0])
            firstFrame = int(row[1])
            lastFrame = int(row[2])

            truthRows.append((clipNum, firstFrame, lastFrame))
        
        colFrameDiffs = []

        # compear each clip
        for i in range(len(truthRows)):
            tClip, tStart, tEnd = truthRows[i]
            rClip, rStart, rEnd = resultRows[i]

            if rClip != tClip:
                print("ERR: Clips Dont Match")
            else:
                frameDiff = abs(tStart - rStart) + abs(tEnd - rEnd)
                colFrameDiffs.append((rClip, frameDiff))
        
        allFrameDiffs.append(colFrameDiffs)
        allCompressions.append((collectionName, colCompressions))  # compression graph
        print("Compeared:", testName, collectionName)


    # create list of lines to write to text file
    lineListA = []
        
    # calculate mean absolute error (sum of all differences / number of clips)
    totalMAE = 0
    numberOfClips = 0
    # setting a cutoff diff so that it doesn't completly distort MAE
    cutoffDiff = 20

    for i in range(len(collections)):
        cfd = allFrameDiffs[i]
        collection = collections[i]
        colMAE = 0

        colLines = []

        for clip, diff in cfd:
            if diff < cutoffDiff:
                colMAE += diff
            else:
                colMAE += cutoffDiff

            colLines.append("  clip" + str(clip) + ": " + str(diff))
        
        totalMAE += colMAE
        numberOfClips += len(cfd)
        colMAE = colMAE / len(cfd)

        lineListA.extend(["\n", collection + " MAE: " + str(colMAE)])
        lineListA.extend(colLines)

    totalMAE = totalMAE / numberOfClips


    # calcuate the accuracies with varying error tollerence
    lineListB = []
    accuracyPoints = []

    lineListB.append("Accuracy within error tollerence:")

    for threshold in range(0, 11):
        numCorrect = 0
        numIncorrect = 0

        for cfd in allFrameDiffs:
            for clip, diff in cfd:
                # decide if clip was correct within the given acceptable frame difference
                if diff <= threshold:
                    numCorrect += 1
                else:
                    numIncorrect += 1
        
        accuracy = round((numCorrect / (numCorrect + numIncorrect)) * 100, 2)

        accuracyPoints.append((threshold, accuracy))
        lineListB.append("  " + str(threshold) + " frames: " + str(accuracy))
    
    # create the final list of output lines
    lineList = []
    lineList.extend([testName, "\n", "Total MAE: " + str(totalMAE), "\n"])
    lineList.extend(lineListB)
    lineList.extend(lineListA)
    
    # write lines to outputFile
    outputFile = open(outputFilePath, "w")
    for line in lineList:
        outputFile.write(line)
        outputFile.write("\n")
    outputFile.close()

    # create accuracy/tollerence plot
    xPoints = np.array([p[0] for p in accuracyPoints])
    yPoints = np.array([p[1] for p in accuracyPoints])

    fig = plt.figure(figsize=(16,9))
    plt.plot(xPoints, yPoints)

    plt.xlabel('Frame Error Tollerence')
    plt.ylabel('Accuracy %')

    plt.grid(b=True, which='both', linestyle='--')

    plt.xticks(xPoints)    
    plt.yticks([0,10,20,30,40,50,60,70,80,90,100])

    plt.show()
    fig.savefig('C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\ContactTests\\' + testName + '\\' + testName + '_accuracy.png', dpi=fig.dpi)

    # create a compression plot
    xPoints = []
    yPoints = []
    for col, colCompressions in allCompressions:
        for compression in colCompressions:
            xPoints.append(col)
            yPoints.append(compression)
    
    fig = plt.figure(figsize=(16,9))
    plt.scatter(xPoints, yPoints)

    plt.xlabel('Collection')
    plt.ylabel('Radius of Contact %')

    plt.grid(b=True, which='both', linestyle='--')

    plt.xticks(xPoints)

    plt.show()
    fig.savefig('C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\ContactTests\\' + testName + '\\' + testName + '_compression4.png', dpi=fig.dpi)


# RUN

collections = [
    "collection_1",
    "not_hitting_wall",
    "person_in_frame",
    "position_close_tilted",
    "position_far_low",
    "position_mid_tilted",
    "regular",
    "slow_lob_shot",
    "tight_angle"
]

test = "test14"

for col in collections:
    genCollectionResults(col, test)

compearResult(test, collections)