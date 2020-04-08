import cv2
import math
import numpy as np
import statistics as stats


def collectData(frame, bgSubMOG, trackPoints, lastSeenBall, linePoints):
    '''
    Uses colour thresholding to find the outline contour in the provided frame.
    Uses foreground extraction to find the ball contour in the provided frame.
    Updates video data stores with detected object data.

    :param frame: the current video frame to collect data from
    :param bgSubMOG: the current open cv background subtractor object
    :param trackPoints: the list of each frames detected ball center and radius
    :param lastSeenBall: the (x,y) coordinates of the centre of the ball in the last frame
    :param linePoints: the list of each frames line contour object
    :returns: the operated image and updated video data
    '''
    # generate output image base
    height, width = frame.shape[:2]
    outputFrame = np.zeros((height,width), np.uint8)

    # threshold on the lines red color
    lowColor = (0,0,75)
    highColor = (50,50,135)
    mask = cv2.inRange(frame, lowColor, highColor)

    # remove noise and join line segments
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel,iterations=4)
    mask = cv2.erode(mask, kernel, iterations=4)

    # retrive the threshold mask contours
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 2:
        contours = contours[0]
    else:
        contours = contours[1]

    # find contour with largest horizontail span indicating the outlines contour
    largestSpan = 0
    largestSpanCon = None
    
    for c in contours:
        leftmost = (c[c[:,:,0].argmin()][0])[0]
        rightmost = (c[c[:,:,0].argmax()][0])[0]
        span = abs(leftmost - rightmost)

        if span > largestSpan:
            largestSpan = span
            largestSpanCon = c
    
    linePoints.append(largestSpanCon)
    
    # draw outline contour on output image
    if len(contours) > 0:
        cv2.drawContours(outputFrame, [largestSpanCon], -1, 128, -1)
    
    # blur and convert frame to grayscale
    frameBlurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frameGray = cv2.cvtColor(frameBlurred, cv2.COLOR_BGR2GRAY)

    # create foreground mask and remove noise and join non-ball objects
    mask = bgSubMOG.apply(frameGray)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel,iterations=4)
    mask = cv2.erode(mask, kernel, iterations=4)

    # find contours in the foreground mask and filter to find ball contour
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    if len(contours) > 0:

        # filter using a min/max ball radius
        possibleBallCons = []
        for con in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(con)
            if radius > 3 and radius < 10:
                possibleBallCons.append(con)

        # find the contour closest to the last known ball
        if len(possibleBallCons) > 0:
            found = False
            closestCon = None
            smallestDelta = 200
            nextBall = (-1,-1)

            lastX, lastY = lastSeenBall
            
            # calculate the centre for each possible ball contour
            for con in possibleBallCons:
                M = cv2.moments(con)
                x = int(M["m10"] / M["m00"]) 
                y = int(M["m01"] / M["m00"])

                delta = math.sqrt(((x - lastX)**2) + ((y - lastY)**2))

                if delta < smallestDelta or lastSeenBall == (-1,-1):
                    smallestDelta = delta
                    closestCon = con
                    nextBall = (x,y)
                    found = True
            
            # draw ball contour if contour found within distance limit
            if found:
                lastSeenBall = nextBall

                ((x, y), radius) = cv2.minEnclosingCircle(closestCon)
                cv2.circle(outputFrame, (int(x), int(y)), int(radius), 255, -1)
                trackPoints.append((int(x), int(y), int(radius)))

            # use 'not detected' dummy data if ball contour not found
            else:
                trackPoints.append((-1,-1,0))
        else:
            trackPoints.append((-1,-1,0))
    else:
        trackPoints.append((-1,-1,0))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(outputFrame, "Collecting Data", (20,70), font, 2, 255, 2, cv2.LINE_AA)

    return (outputFrame, bgSubMOG, trackPoints, lastSeenBall, linePoints)


def calcContactFrames(gradRatePoints, deltaPoints):
    '''
    Caclulates the frame numbers of the the video where the ball makes contact with the wall.
    Also calculates the percentage of the balls radius is in contact with the wall for each frame of contact.

    :params gradRatePoints: the list of the rate of change of gradient of the balls flight in each frame
    :params deltaPoints: the list of the stright line distance between current and last frames ball centre
    :returns: the list of pairs of (frame number, radius percentage of contact) where the ball contacts the wall
    '''
    # initalise variables in case no contact is detected
    contactFrames = []
    threshold = 0.08
    contactIndex = -1
    contactGradRate = 0

    # find position when contact happened as point at which gradient rate exceeds a threshold
    for i in range(len(gradRatePoints)):
        gradRate = gradRatePoints[i]

        if gradRate != None and contactGradRate == 0:
            if gradRate > threshold:
                contactIndex = i - 1
                contactGradRate = gradRate

    if contactGradRate != 0:
        # estimate the number of contact frames based on change on trajectory at point of contact
        if contactGradRate < 0.095:
            contactFrames = [contactIndex - 1, contactIndex, contactIndex + 1, contactIndex + 2]
        elif contactGradRate >= 0.095 and contactGradRate < 0.14:
            contactFrames = [contactIndex - 1, contactIndex, contactIndex + 1]
        elif contactGradRate >= 0.14:
            contactFrames = [contactIndex - 1, contactIndex]

        # calculate average ball speed before and after contact
        if contactIndex - 11 < 0:
            deltaBefore = stats.median(deltaPoints[0 : contactIndex - 1])
        else:
            deltaBefore = stats.median(deltaPoints[contactIndex - 11 : contactIndex - 1])

        if contactIndex + 11 > len(deltaPoints) - 1:
            deltaAfter = stats.median(deltaPoints[contactIndex + 1:])
        else:
            deltaAfter = stats.median(deltaPoints[contactIndex + 1: contactIndex + 11])

        # estimate percentage of radius of ball in contact with the wall within set thresholds
        compDistance = (((deltaBefore**2) - (deltaAfter**2)) * (1 + contactGradRate)) / len(contactFrames)
        contactPercent = ((compDistance + 20) / 60) * 100

        maxContactPercent = 95
        minContactPercent = 50
        if contactPercent > maxContactPercent:
            contactPercent = maxContactPercent
        if contactPercent < minContactPercent:
            contactPercent = minContactPercent

        # set ball contact percentage in each frame of contact frames
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


def calcPointRateGrad(gradPoints):
    '''
    Calculate the rate at which the gradient of the ball is changing in each frame.

    :param gradPoints: the list of the gradient of the balls flight in each frame
    :returns: the list of the rate of change of gradient of the balls flight in each frame
    '''
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


def calcPointGrad(predPoints):
    '''
    Calculate the gradient of the balls trajectory in each frame.

    :param predPoints: the list of each frames predicted ball center and radius
    :returns: the list of the gradient of the balls flight in each frame
    '''
    gradPoints = []

    gradPoints.append(None)
    gradPoints.append(None)
    gradPoints.append(None)

    for i in range(3, len(predPoints)):
        prevX, prevY = predPoints[i-3][:2]
        currX, currY = predPoints[i][:2]

        # check for divide by 0 error and that both frames are of detected balls
        if (currX - prevX) != 0 and prevX != -1 and currX != -1:
            m = (currY - prevY)/(currX - prevX)
            gradPoints.append(m)
        else:
            gradPoints.append(None)
    
    return gradPoints


def calcDeltaPoints(predPoints):
    '''
    Calculate the 'speed' of the ball in each frame.

    :param predPoints: the list of each frames predicted ball center and radius
    :returns: the list of the stright line distance between current and last frames ball centre
    '''
    deltaPoints = []
    deltaPoints.append(0)

    for i in range(1, len(predPoints)):
        prevX, prevY = predPoints[i-1][:2]
        currX, currY = predPoints[i][:2]

        delta = round(math.sqrt(((currX - prevX)**2) + ((currY - prevY)**2)), 3)

        # if delta is 0 its because frame was repeated so use last frames delta
        if delta == 0 or prevX == -1:
            delta = deltaPoints[i-1]

        deltaPoints.append(delta)
    
    return deltaPoints


def removeListNoise(myList):
    '''
    Averages each item in the list with its two neighbours 'smoothing' its values.

    :param myList: the list of numbers
    :returns: the input list with noise removed
    '''
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


def expandTrackGaps(trackPoints, linePoints):
    '''
    If the ball is not detected in either of a frames neighbouring frames set the ball as not detected in this frame too.
    This is because the ball will only be half detected as it is partially occluded by the line.
    Overwrite the detected line contour for these frames with previous line contour due to the balls occlison distorting colour thresholding.

    :param trackPoints: the list of each frames detected ball center and radius
    :param linePoints: the list of each frames line contour object
    :returns: the pair of input lists with their missing value gaps expanded
    '''
    # initalise the new line and track data lists
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
    '''
    Predicts the ball centre and radius when its been detected originally using linear extrapolation.

    :param trackPoints: the list of each frames detected ball center and radius
    :returns: the list of each frames predicted ball center and radius
    '''
    # find the gaps of ball detection
    missingSections = [[]]
    sectionCount = 0

    for i in range(1, len(trackPoints) - 1):
        x, y, r = trackPoints[i]
        pX = trackPoints[i - 1][0]
        nX = trackPoints[i + 1][0]

        # adds missing points to gap
        if x < 0:
            missingSections[sectionCount].append((x,y,r,i))
        # adds real value far to end of gap and increments section count
        elif pX < 0:
            missingSections[sectionCount].append((x,y,r,i))
            sectionCount += 1
        # adds real value at start of gap
        elif nX < 0:
            missingSections.append([])
            missingSections[sectionCount].append((x,y,r,i))
    
    # predict ball centre and radius in the missing sections, excluding the first points where ball has not yet been found
    for section in missingSections:
        if section[0][0] != -1 and section[-1][0] != -1:

            startX, startY, startR = section[0][:3]
            endX, endY, endR = section[-1][:3]
            numMissing = len(section) - 2

            xStep = (endX - startX) / (numMissing + 1)
            yStep = (endY - startY) / (numMissing + 1)

            missingR = max(startR, endR)
            
            for i in range(1, len(section) - 1):
                pos = section[i][3]
                missingX = int(startX + (i * xStep))
                missingY = int(startY + (i * yStep))
                section[i] = (missingX, missingY, missingR)
                
                trackPoints[pos] = section[i]
                      
    return trackPoints