import numpy as np
import os
import math
import cv2
import v9_calculations as calc
import csv
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import randint


# returns the contact frames for a given path
def getOutProb(clipPath):
    trackPoints = []
    predPoints = []
    linePoints = []
    deltaPoints = []
    gradPoints = []
    rateGradPoints = []

    contactFrames = []
    lastSeenBall = (-1,-1)

    contactPrints = []
    outProb = 0

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

    gradPoints = calc.calcPointGrad(predPoints)
    gradPoints = calc.removeListNoise(gradPoints)
    rateGradPoints = calc.calcPointRateGrad(gradPoints)

    deltaPoints = calc.calcDeltaPoints(predPoints)

    # take compression distance 
    contactFrames = calc.calcContactFrames(rateGradPoints, deltaPoints)

    contactPrints = []
    outProb = 0

    height = 1080
    width = 1920

    for frameIndex, radiusPercent in contactFrames:
        ballData = predPoints[frameIndex]
        lineData = linePoints[frameIndex]

        # create then contactPrint mask
        ballCenter = ballData[:2]
        ballRadius = ballData[2]
        ballPrintRadius = int(math.ceil(ballRadius * (radiusPercent / 100)))

        ballPrintMask = np.zeros((height, width), np.uint8)
        cv2.circle(ballPrintMask, ballCenter, ballPrintRadius, 255, -1)

        # add current ballPrintMask to list to be used for remaining contactFrames
        contactPrints.append(ballPrintMask)

        # create the contactMask "so far"
        contactMask = np.zeros((height, width), np.uint8)
        for mask in contactPrints:
            contactMask = cv2.add(mask, contactMask)
        
        # join up indevidual prints into a single print
        kernel = np.ones((7,7), np.uint8)
        contactMask = cv2.dilate(contactMask, kernel, iterations=2)
        contactMask = cv2.erode(contactMask, kernel, iterations=2)

        # create line mask
        lineMask = np.zeros((height, width), np.uint8)
        cv2.drawContours(lineMask, [lineData], -1, 255, 1)

        probMask, newOutProb = calc.probOut(contactMask, lineMask)

        # update outProb if the new frame probability is larger
        if newOutProb > outProb:
            outProb = newOutProb

    return outProb



def genCollectionResults(collectionName, testName):
    clipDirPath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\Footage\\' + collectionName + '\\'
    resultDirPath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\DecisionTests\\' + testName + '\\csv\\'
    resultFilePath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\DecisionTests\\' + testName + '\\csv\\' + collectionName + '.csv'


    # get every clip file name in clipDirPath
    clipFiles = []

    for r, d, f in os.walk(clipDirPath):
        for clip in f:
            clipFiles.append(str(clip))

    
    # stores probabilities of clips
    clipProbs = []

    for clip in clipFiles:
        print("Calculating Prob:", collectionName, clip)
        clipPath = clipDirPath + clip

        outProb = getOutProb(clipPath) 
        clipProbs.append(outProb)
        print("Prob = ", outProb)
        

    # generate csv rows
    rows = []

    for i in range(len(clipFiles)):
        if clipFiles[i][5] != '.':
            clipNum = int(clipFiles[i][4:6])
        else:
            clipNum = int(clipFiles[i][4])
        
        prob = clipProbs[i]

        rows.append([clipNum, prob])  # compression graph
    
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




# generate truth value from collected responses
def genTruth(collections):
    truthDir = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\DecisionTests\\truth\\'
    responseDir = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\ParticipantSurvay\\Responses\\Formatted\\'

    # get every response
    responses = []

    for r, d, f in os.walk(responseDir):
        for response in f:
            responses.append(str(response))
    
    truths = []

    # go through each response csv
    for response in responses:

        responseTruth = []
        collectionCount = 0
        currCollection = []

        for line in open(str(responseDir + response)):
            row = line.split(',')

            clipNum = row[0]
            decision = row[1]

            # start of a new set increment the collectionCount
            if 'S' in clipNum:
                if len(currCollection) != 0:
                    responseTruth.append(currCollection)
                    collectionCount += 1
                    currCollection = []
            else:
                if decision.lower()[0] == 'i':
                    decision = 0
                else:
                    decision = 1
                currCollection.append(decision)

        # add last collection results responseTruth and add responseTruth to list of all responseTruths
        responseTruth.append(currCollection)
        truths.append(responseTruth)
    
    results = []

    # calculate the probability for each clip of each collection
    for i in range(len(truths[0])):
        collectionProbs = []
        for j in range(len(truths[0][i])):
            clipSum = 0

            for responseTruth in truths:
                clipSum += responseTruth[i][j]

            prob = round(clipSum / len(truths), 2)
            collectionProbs.append(prob)
        results.append(collectionProbs)
    
    
    # write to csv file for each collection
    for i in range(len(results)):
        collectionName = collections[i]
        collectionResults = results[i]

        rows = []

        for j in range(len(collectionResults)):
            rows.append([collectionResults[j]])
        
        # generate directory
        try:
            os.makedirs(truthDir)
        except OSError:
            print("Truth Dir Exists")
        else:
            print("Truth Dir Created")

        truthFile = str(truthDir + collectionName + ".csv")
        # write rows to a csv
        with open(truthFile, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)


# compear a generated csv with the truth csv
def compearResult(testName, collections):
    outputFilePath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\DecisionTests\\' + testName + '\\'  + testName + '_results.txt'
    allProbDiffs = []

    for collectionName in collections:
        resultFilePath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\DecisionTests\\' + testName + '\\csv\\' + collectionName + '.csv'
        truthFilePath = 'C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\DecisionTests\\truth\\' + collectionName + '.csv'

        # read truth and test csv for configuration
        truthRows = []
        resultRows = []
        colProbs = []

        for line in open(resultFilePath):
            row = line.split(',')
            clipNum = int(row[0])
            prob = float(row[1])
            
            colProbs.append(prob)
            resultRows.append((clipNum, prob)) 
        
        for line in open(truthFilePath):
            row = line.split(',')
            prob = float(row[0])

            truthRows.append(prob)
        
        colProbDiffs = []

        # compear each clip
        for i in range(len(truthRows)):
            tProb = truthRows[i]
            rClip, rProb = resultRows[i]
       
            probDiff = abs(tProb - rProb)
            
            # see if overall decision agree's the the majority of participants
            if (tProb > 0.5 and rProb > 0.5) or (tProb < 0.5 and rProb < 0.5):
                decision = "Correct"
            elif (tProb > 0.5 and rProb < 0.5) or (tProb < 0.5 and rProb > 0.5):
                decision = "Incorrect"
            else:
                decision = "Split"

            colProbDiffs.append((rClip, probDiff, decision))
        
        allProbDiffs.append(colProbDiffs)
        print("Compeared:", testName, collectionName)


    # create list of lines to write to text file
    lineListA = []
        
    # calculate mean absolute error (sum of all differences / number of clips)
    totalMAE = 0
    totalDecAcc = 0
    numberOfClips = 0

    for i in range(len(collections)):
        cpd = allProbDiffs[i]
        collection = collections[i]
        colMAE = 0
        colDecAcc = 0

        colLines = []

        for clip, diff, decision in cpd:
            colMAE += diff

            if decision == "Correct" or decision == "Split":
                colDecAcc += 1

            colLines.append("  clip" + str(clip) + ": " + str(round(diff, 4)) + " - " + decision)
        
        totalMAE += colMAE
        totalDecAcc += colDecAcc
        numberOfClips += len(cpd)
        colMAE = colMAE / len(cpd)
        colDecAcc = colDecAcc / len(cpd)

        lineListA.extend(["\n", collection + " MAE: " + str(round(colMAE, 4)) + " - " + str(round(colDecAcc * 100, 2))])
        lineListA.extend(colLines)

    totalMAE = totalMAE / numberOfClips
    totalDecAcc = (totalDecAcc / numberOfClips) * 100


    # calcuate the accuracies with varying error tollerence
    lineListB = []
    accuracyPoints = []

    lineListB.append("Accuracy within error tollerence:")

    for threshold in range(0, 5):
        numCorrect = 0
        numIncorrect = 0
        threshold = round(0.1 * threshold, 2)

        for cpd in allProbDiffs:
            for clip, diff, decision in cpd:
                # decide if clip was correct within the given acceptable prob difference
                if diff <= threshold:
                    numCorrect += 1
                else:
                    numIncorrect += 1
        
        accuracy = round((numCorrect / (numCorrect + numIncorrect)) * 100, 2)

        accuracyPoints.append((threshold, accuracy))
        lineListB.append("  " + str(threshold) + " prob tollerence: " + str(round(accuracy, 2)))
    
    # create the final list of output lines
    lineList = []
    lineList.extend([testName, "\n", "Total MAE: " + str(round(totalMAE, 4)), "Total Decision Accuracy: " + str(round(totalDecAcc, 2)), "\n"])
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

    plt.xlabel('Decision Probability Error Tollerence')
    plt.ylabel('Accuracy %')

    plt.grid(b=True, which='both', linestyle='--')

    plt.xticks(xPoints)    
    plt.yticks([50,55,60,65,70,75,80,85,90,95,100])

    plt.show()
    fig.savefig('C:\\Users\\Luke\\Documents\\Google Drive\\University\\A Part III Project\\SquashVision\\Project\\TestResults\\DecisionTests\\' + testName + '\\' + testName + '_accuracy.png', dpi=fig.dpi)



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

test = "test2"

# genTruth(collections)

for col in collections:
    genCollectionResults(col, test)

compearResult(test, collections)