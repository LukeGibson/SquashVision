import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import seaborn as sns
import statistics as stats
import numba as nb


# fast function to fill lineMask above the line
@nb.jit
def convertLineMask(lineMask):
    for x in range(len(lineMask)):
        col = lineMask[x]
        lineMinY = -1

        for y in range(len(col)):
            # find the start of the line
            if lineMask[x][-y] != 0:
                lineMinY = y
                break
        
        # add probabilities 
        if lineMinY != -1:
            under = [0] * (lineMinY - 3)
            outer = [80] * 3
            inner = [160] * 2
            over = [240] * (len(col) - lineMinY - 2)

            newCol = over + inner + outer + under
            lineMask[x] = newCol

    return lineMask


# given the 2 masks sums them to find values of contact - allows for probability of out to be calculated
def probOut(contactMask, lineMask):
    # convert line mask and add probability
    lineMask2 = cv2.transpose(lineMask)
    lineMask2 = convertLineMask(lineMask2)
    lineMask2 = cv2.transpose(lineMask2)

    # add contactMask probability
    kernelSmall = np.ones((3,3), np.uint8)
    contactMaskInner = cv2.erode(contactMask, kernelSmall)
    kernelLarge = np.ones((5,5), np.uint8)
    contactMaskOuter = cv2.dilate(contactMask, kernelLarge)

    contactOuterSize = np.count_nonzero(np.array(contactMaskOuter).flatten() == 255)
    contactSize = np.count_nonzero(np.array(contactMask).flatten() == 255)

    # check Inner maskis not dilated to nothing
    contactInnerSize = np.count_nonzero(np.array(contactMaskInner).flatten() == 255)
    if contactInnerSize < 5:
        print("ContactMaskInner was erouded to much - contained", contactInnerSize, "pixels")
        contactMaskInner = contactMask
        contactInnerSize = np.count_nonzero(np.array(contactMaskInner).flatten() == 255)
        print("ContactMaskInner set to contactMask - contains", contactInnerSize, "pixels")

    # convert contactMasks values of 255 to 240
    contactMask = cv2.addWeighted(contactMask, (8/17), contactMask, (8/17), 0)
    contactMaskInner = cv2.addWeighted(contactMaskInner, (8/17), contactMaskInner, (8/17), 0)
    contactMaskOuter = cv2.addWeighted(contactMaskOuter, (8/17), contactMaskOuter, (8/17), 0)

    # sum together the component contactMask's
    contactMask2 = cv2.addWeighted(contactMaskOuter, 0.5, contactMaskInner, 0.5, 0)
    contactMask2 = cv2.addWeighted(contactMask2, (2/3), contactMask, (1/3), 0)

    probMask = cv2.addWeighted(lineMask2, 0.5, contactMask2, 0.5, 0)


    # use masks to calculate the probability the shot was out
    probMaskFlat = np.array(probMask).flatten()
    maxValue = max(probMaskFlat)
    maxCount = np.count_nonzero(probMaskFlat == maxValue)

    # calculates the min and max probabilty depending on the max value in the probMask
    # set largest number of contact pixels that could create maxValue
    if maxValue <= 120:
        thresholdA = 0
        thresholdB = 0
        trueContactSize = 0
    elif maxValue <= 160:
        thresholdA = 0
        thresholdB = 0.66
        trueContactSize = contactOuterSize - contactSize
    elif maxValue <= 200:
        thresholdA = 0.66
        thresholdB = 0.83
        trueContactSize = contactSize - contactInnerSize
    elif maxValue <= 240:
        thresholdA = 0.83
        thresholdB = 1
        trueContactSize = contactInnerSize
    else:
        print("Prob Calc Error: maxValue > 240")
        thresholdA = -1
        thresholdB = -1    
    
    
    print("MaxValue:", maxValue,"MaxCount:", maxCount, "trueContactSize:", trueContactSize)

    # only calculate the probabilty if thesholdB > 0
    if thresholdB > 0:

        # error check
        if maxCount > trueContactSize:
            print("Prob Calc Error: maxCount > trueContactSize")
            trueContactSize = maxCount

        # the amount probabilty can vary between min and max
        difference = thresholdB - thresholdA
        
        # calculate percentage of inner pixels that are at this max value
        proprotionMaxContact =  maxCount / trueContactSize
        print("proportionMaxContact", proprotionMaxContact)

        # probabilty is closer to thresholdB the greater proportion of the pixels were at the max value
        probability = round(thresholdA + (difference * proprotionMaxContact), 2)
    else:
        probability = 0

    print("prob", probability)

    return (probMask, probability)


# display graphs for a given set of lists
def displayGraphs(yValueList):
    fig1, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 10))
    frameNumbers = list(range(0, len(yValueList[0][0])))

    for i in range(len(yValueList)):
        values, title = yValueList[i]

        if i < 3:
            sns.lineplot(frameNumbers, values, ax=ax[0][i])
            ax[0][i].set(title=title)
        else:
            sns.lineplot(frameNumbers, values, ax=ax[1][i-3])
            ax[1][i-3].set(title=title)

    plt.show()

# uses input linear gradient to find when linear gradient changes
def calcContactFrames(gradRatePoints, deltaPoints):
    contactFrames = []

    threshold = 0.08
    contactIndex = -1
    contactGradRate = 0

    for i in range(len(gradRatePoints)):
        gradRate = gradRatePoints[i]

        if gradRate != None and contactGradRate == 0:
            if gradRate > threshold:
                contactIndex = i - 1 # -1 as in a frame is calculated from a point 2 frames previous
                contactGradRate = gradRate

    if contactGradRate != 0:
        # estimate the number of contact frames based on change on trajectory at point of contact
        if contactGradRate < 0.095:
            contactFrames = [contactIndex - 1, contactIndex, contactIndex + 1, contactIndex + 2]
        elif contactGradRate >= 0.095 and contactGradRate < 0.14:
            contactFrames = [contactIndex - 1, contactIndex, contactIndex + 1]
        elif contactGradRate >= 0.14:
            contactFrames = [contactIndex - 1, contactIndex]

        # calculate speed before and after impact
        if contactIndex - 11 < 0:
            deltaBefore = stats.median(deltaPoints[0 : contactIndex - 1])
        else:
            deltaBefore = stats.median(deltaPoints[contactIndex - 11 : contactIndex - 1])

        if contactIndex + 11 > len(deltaPoints) - 1:
            deltaAfter = stats.median(deltaPoints[contactIndex + 1:])
        else:
            deltaAfter = stats.median(deltaPoints[contactIndex + 1: contactIndex + 11])

        # compression is proportional to the loss of KE over time - more energy lost over shorter time = greater compression
        compDistance = (((deltaBefore**2) - (deltaAfter**2)) * (1 + contactGradRate)) / len(contactFrames)

        # percentage of radius of ball in contact with the wall
        contactPercent = ((compDistance + 20) / 60) * 100

        maxContactPercent = 95
        minContactPercent = 50

        if contactPercent > maxContactPercent:
            contactPercent = maxContactPercent
        if contactPercent < minContactPercent:
            contactPercent = minContactPercent

        # calculate the ballContactPercent in each frame of contactFrames
        if len(contactFrames) == 2:
            contactPercents = [contactPercent, contactPercent * 0.75]
        elif len(contactFrames) == 3:
            contactPercents = [contactPercent * 0.5, contactPercent, contactPercent * 0.75]
        elif len(contactFrames) == 4:
            contactPercents = [contactPercent * 0.5, contactPercent, contactPercent * 0.75, contactPercent * 0.5]

        # create a list of pairs (frame number, radius of ball in contact)
        contactFrames = list(zip(contactFrames, contactPercents))

    else:
        contactFrames = []

    return contactFrames


# calculates the rate of gradient change of the line going into each point
def calcPointRateGrad(gradPoints):
    rateGradPoints = []

    rateGradPoints.append(None)
     
    for i in range(1, len(gradPoints)):
        prevGrad = gradPoints[i-1]
        currGrad = gradPoints[i]

        if prevGrad != None and currGrad != None:
            gradChange = abs(prevGrad - currGrad)
            rateGradPoints.append(gradChange)
        else:
            rateGradPoints.append(None)

    return rateGradPoints


# calculates the gradient of the line going into each point
def calcPointGrad(predPoints):
    gradPoints = []

    gradPoints.append(None)
    gradPoints.append(None)
    gradPoints.append(None)

    for i in range(3, len(predPoints)):
        prevX, prevY = predPoints[i-3][:2]
        currX, currY = predPoints[i][:2]

        # check for divide by 0 error and that both points are of detected balls
        if (currX - prevX) != 0 and prevX != -1 and currX != -1:
            m = (currY - prevY)/(currX - prevX)
            gradPoints.append(m)
        else:
            gradPoints.append(None)
    
    return gradPoints
            

# calculate rate of angle change points
def calcPointRateAngles(anglePoints):
    rateAnglePoints = []

    rateAnglePoints.append(None)
     
    for i in range(1, len(anglePoints)):
        prevAngle = anglePoints[i-1]
        currAngle = anglePoints[i]

        if prevAngle != None and currAngle != None:
            angleChange = abs(prevAngle - currAngle)
            rateAnglePoints.append(angleChange)
        else:
            rateAnglePoints.append(None)

    return rateAnglePoints


# given a list of ball points calculate the angle at each point
def calcPointAngles(predPoints):
    anglePoints = []
    
    # angle at the first point can't be calculated
    anglePoints.append(None)
    anglePoints.append(None)

    # calculate the angle at each point using prev and next point 
    for i in range(2, len(predPoints) - 2):
        prevX, prevY = predPoints[i - 2][:2]
        currX, currY = predPoints[i][:2]
        nextX, nextY = predPoints[i + 2][:2]

        # check all points are of a detected ball
        if -1 in [prevX, currX, nextX]:
            angle = None
        # use the cosine rule to calculate the angle at current point: cos(C) = (a^2 + b^2 - c^2) / 2ab
        else:
            # calculate vectors
            a = math.sqrt(((currX - nextX)**2) + ((currY - nextY)**2))
            b = math.sqrt(((currX - prevX)**2) + ((currY - prevY)**2))
            c = math.sqrt(((prevX - nextX)**2) + ((prevY - nextY)**2))
            
            # calculate angle
            if a > 0 and b > 0:
                cosC = ((a**2) + (b**2) - (c**2)) / (2 * a * b)
                cosC = np.clip(cosC, -1, 1)
                angle = round(math.degrees(math.acos(cosC)), 3)
            else:
                angle = None

        anglePoints.append(angle)
    
    # angle at the last point can't be calculated
    anglePoints.append(None)
    anglePoints.append(None)

    return anglePoints


# calculate delta points
def calcDeltaPoints(predPoints):
    deltaPoints = []

    # first change must be 0
    deltaPoints.append(0)

    for i in range(1, len(predPoints)):
        prevX, prevY = predPoints[i-1][:2]
        currX, currY = predPoints[i][:2]

        delta = round(math.sqrt(((currX - prevX)**2) + ((currY - prevY)**2)), 3)

        # if delta is 0 its because frame was repeated - use last frames delta
        if delta == 0 or prevX == -1:
            delta = deltaPoints[i-1]

        deltaPoints.append(delta)
    
    return deltaPoints


# remove noise from list
def removeListNoise(myList):
    roundedList = []

    roundedList.append(myList[0])

    for i in range(1, len(myList) - 1):
        roundedItem = 0
        count = 0

        if myList[i-1] != None:
            roundedItem += myList[i-1]
            count += 1
        if myList[i] != None:
            roundedItem += myList[i]
            count += 1
        if myList[i+1] != None:
            roundedItem += myList[i+1]
            count += 1

        if count > 0:
            roundedItem = roundedItem / count
            roundedList.append(roundedItem)
        else:
            roundedList.append(None)

    roundedList.append(myList[-1])

    return roundedList


# expand the track gaps as ball will distort line detection when half over and overwrite the linePoints
def expandTrackGaps(trackPoints, linePoints):
    cleanTrackPoints = []
    cleanTrackPoints.append(trackPoints[0])
    cleanTrackPoints.append(trackPoints[0])

    cleanLinePoints = []
    cleanLinePoints.append(linePoints[0])
    cleanLinePoints.append(linePoints[0])

    # if either neighbouring point doesn't have a ball detection assume the detection in current point is poor
    for i in range(2, len(trackPoints) - 2):
        if trackPoints[i-1] != (-1,-1,0) and trackPoints[i+1] != (-1,-1,0) and trackPoints[i+2] != (-1,-1,0) and trackPoints[i-2] != (-1,-1,0):
            cleanTrackPoints.append(trackPoints[i])
            cleanLinePoints.append(linePoints[i])
        else:
            cleanTrackPoints.append((-1,-1,0))
            cleanLinePoints.append(cleanLinePoints[i-1])
    
    cleanTrackPoints.append(trackPoints[-1])
    cleanTrackPoints.append(trackPoints[-1])

    cleanLinePoints.append(linePoints[-1])
    cleanLinePoints.append(linePoints[-1])

    return cleanTrackPoints, cleanLinePoints


def fillTrackGaps(trackPoints):
    # find all sections with a missing point
    missingSections = [[]]
    sectionCount = 0

    for i in range(1, len(trackPoints) - 1):
        x, y, r = trackPoints[i]
        pX, pY, pR = trackPoints[i - 1]
        nX, nY, nR = trackPoints[i + 1]

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

            # setting missing point radius as the largest of start and end radius
            missingR = max(startR, endR)
            
            # calculate ball position at even spacing for missing values (assumes a stright line)
            for i in range(1, len(section) - 1):
                pos = section[i][3]
                missingX = int(startX + (i * xStep))
                missingY = int(startY + (i * yStep))
                section[i] = (missingX, missingY, missingR)
                
                # rewrite missing point in full list
                trackPoints[pos] = section[i]
                      
    return trackPoints