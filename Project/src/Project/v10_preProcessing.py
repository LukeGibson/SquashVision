import cv2
import math
import numpy as np
import statistics as stats


def generateTrackVid(frame, bgSubMOG, trackPoints, lastSeenBall, linePoints):
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
                cv2.circle(outputFrame, (int(x), int(y)), int(radius), 255, -1)

                trackPoints.append((int(x), int(y), int(radius)))
            else:
                trackPoints.append((-1,-1,0))
        else:
            trackPoints.append((-1,-1,0))
    else:
        trackPoints.append((-1,-1,0))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(outputFrame, "Collecting Data", (20,70), font, 2, 255, 2, cv2.LINE_AA)

    return (outputFrame, bgSubMOG, trackPoints, lastSeenBall, linePoints)


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