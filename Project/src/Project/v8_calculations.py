import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import seaborn as sns


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
                contactIndex = i - 0 # -1 as in a frame is calculated from a point 2 frames previous
                contactGradRate = gradRate

    if contactGradRate != 0:

        # estimate the number of contact frames based on change on trajectory at point of contact
        if contactGradRate < 0.1:
            contactFrames = [contactIndex - 1, contactIndex, contactIndex + 1, contactIndex + 2]
        elif contactGradRate >= 0.1 and contactGradRate < 0.16:
            contactFrames = [contactIndex - 1, contactIndex, contactIndex + 1]
        elif contactGradRate >= 0.16:
            contactFrames = [contactIndex, contactIndex + 1]      

    else:
        contactFrames = []

    print("Contact Frames:", contactFrames)

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
        if delta == 0:
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
    cleanTrackPoints.append(trackPoints[1])

    cleanLinePoints = []
    cleanLinePoints.append(linePoints[0])
    cleanLinePoints.append(linePoints[1])

    # if either neighbouring point doesn't have a ball detection assume the detection in current point is poor
    for i in range(2, len(trackPoints) - 2):
        if trackPoints[i-1] != (-1,-1,0) and trackPoints[i+1] != (-1,-1,0) and trackPoints[i+2] != (-1,-1,0) and trackPoints[i-2] != (-1,-1,0):
            cleanTrackPoints.append(trackPoints[i])
            cleanLinePoints.append(linePoints[i])
        else:
            cleanTrackPoints.append((-1,-1,0))
            cleanLinePoints.append(cleanLinePoints[i-1])
    
    cleanTrackPoints.append(trackPoints[-2])
    cleanTrackPoints.append(trackPoints[-1])

    cleanLinePoints.append(linePoints[-2])
    cleanLinePoints.append(linePoints[-1])

    return cleanTrackPoints, cleanLinePoints


def fillTrackGaps(trackPoints):
    # find all sections with a missing point
    missingSections = [[]]
    sectionCount = 0

    for i in range(len(trackPoints) - 1):
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
