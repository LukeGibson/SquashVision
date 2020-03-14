import numpy as np
import os
import math
import cv2
import v7_calculations as calc
import csv
from pathlib import Path


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
    currCapture = cv2.VideoCapture(clipPath)

    trackPoints = calc.expandTrackGaps(trackPoints)
    predPoints = calc.fillTrackGaps(trackPoints)

    gradPoints = calc.calcPointGrad(predPoints)
    gradPoints = calc.removeListNoise(gradPoints)
    rateGradPoints = calc.calcPointRateGrad(gradPoints)
    contactFrames = calc.calcContactFrames4(rateGradPoints, deltaPoints)

    return contactFrames



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

    for clip in clipFiles:
        print("Calculating Contact:", collectionName, clip)
        clipPath = clipDirPath + clip
        contactFrames = getContactFrames(clipPath)
        clipContactFrames.append(contactFrames)

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
        else:
            frameStart = clipContactFrames[i][0]
            frameEnd = clipContactFrames[i][-1]

        rows.append([clipNum, frameStart, frameEnd])
    
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
    outputFilePath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\ContactTests\\' + testName + '\\results.txt'
    allQualityCounts = []

    for collectionName in collections:
        resultFilePath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\ContactTests\\' + testName + '\\csv\\' + collectionName + '.csv'
        truthFilePath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\ContactTests\\truth\\' + 'truth_' + collectionName + '.csv'

        # read truth and test csv for configuration
        truthRows = []
        resultRows = []

        for line in open(resultFilePath):
            row = line.split(',')
            clipNum = int(row[0])
            firstFrame = int(row[1])
            lastFrame = int(row[2][:-1])

            resultRows.append((clipNum, firstFrame, lastFrame))
        
        for line in open(truthFilePath):
            row = line.split(',')
            clipNum = int(row[0])
            firstFrame = int(row[1])
            lastFrame = int(row[2][:-1])

            truthRows.append((clipNum, firstFrame, lastFrame))
        
        # sorts clips of Good, Poor, Wrong quality ratings
        qualityCounts = [
            [],
            [],
            [],
        ]

        # compear each clip
        for i in range(len(truthRows)):
            tClip, tStart, tEnd = truthRows[i]
            rClip, rStart, rEnd = resultRows[i]

            if rClip != tClip:
                print("ERR: Clips Dont Match")
            else:
                # condition ball didnt hit the wall
                if tStart == 0 and tEnd == 0:
                    if rStart != 0 and rEnd != 0:
                        qualityCounts[2].append(rClip)
                    else:
                        qualityCounts[0].append(rClip)
                else:
                    startDiff = abs(tStart - rStart)
                    endDiff = abs(tEnd - rEnd)
                    # Good if difference between start and end isn't more than 1
                    if startDiff < 2 and endDiff < 2:
                        qualityCounts[0].append(rClip)
                    # Poor if differnece between start and end isn't more than 2
                    elif startDiff < 3 and endDiff < 3:
                        qualityCounts[1].append(rClip)
                    # Wrong otherwise
                    else:
                        qualityCounts[2].append(rClip)

        allQualityCounts.append(qualityCounts)
        print("Compeared:", testName, collectionName)
    
    # calculate total accuracy percentages
    totalGood = 0
    totalPoor = 0
    totalWrong = 0

    for qCounts in allQualityCounts:
        totalGood += len(qCounts[0])
        totalPoor += len(qCounts[1])
        totalWrong += len(qCounts[2])
    
    total = totalGood + totalPoor + totalWrong
    perGood = round((totalGood / total) * 100, 2)
    perPoor = round((totalPoor / total) * 100, 2)
    perWrong = round((totalWrong / total) * 100, 2)

    # create list of lines to write to text file
    lineList = []
    lineList.append("Total Percentages")
    lineList.append("  Good: " + str(perGood))
    lineList.append("  Poor: " + str(perPoor))
    lineList.append("  Wrong: " + str(perWrong))
    lineList.append("\n")

    for i in range(len(collections)):
        lineList.append(collections[i] + '')

        qCounts = allQualityCounts[i]
        lineList.append("  Good: " + str(qCounts[0])[1:-1])
        lineList.append("  Poor: " + str(qCounts[1])[1:-1])
        lineList.append("  Wrong: " + str(qCounts[2])[1:-1])
        lineList.append("\n")
    
    # write lines to outputFile
    outputFile = open(outputFilePath, "w")
    for line in lineList:
        outputFile.write(line)
        outputFile.write("\n")
    outputFile.close()




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

test = "test5"

for col in collections:
    genCollectionResults(col, test)

compearResult(test, collections)
