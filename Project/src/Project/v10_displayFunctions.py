import cv2
import numpy as np
import math
import v10_postProcessing as post


def decisionVid(frame, trackPoints, trackPredPoints, linePoints, gradPoints, rateGradPoints, deltaPoints, frameIndex, contactFrames, contactPrints, outProb, vidOp):
    # get frame dimensions
    height, width = frame.shape[:2]

    # get frame line and ball data
    ballData = trackPredPoints[frameIndex]
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

    # create the ball mask
    ballMask = np.zeros((height, width), np.uint8)
    cv2.circle(ballMask, ballData[:2], ballData[2], 255, 1)


    # Find if ball is out
    if frameIndex in [i[0] for i in contactFrames]:
        # calculate the probability the ball was out
        probMask, newOutProb, maxValue = post.probOut(contactMask, lineMask)

        # update outProb if the new frame probability is larger
        if newOutProb > outProb:
            outProb = newOutProb
    else:
        probMask = cv2.addWeighted(lineMask, 0.5, ballMask, 0.5, 0)
        maxValue = 120

    # create colored masks for output
    lineMaskCol = np.zeros((height, width, 3), np.uint8)
    cv2.drawContours(lineMaskCol, [lineData], -1, (255,0,0), 1)

    ballMaskCol = np.zeros((height, width, 3), np.uint8)
    cv2.circle(ballMaskCol, ballData[:2], ballData[2], (0,255,0), 1)

    # write text on output
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Set output depending on current selected operation
    if vidOp == "Make Decision":
        # add colered masks together
        output = cv2.add(ballMaskCol, lineMaskCol)
        output = cv2.add(output, contactMaskCol, 1)

        if ballContact:
            cv2.putText(output, "Contact: True", (20,70), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(output, "Contact: False", (20,70), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
            
        if outProb >= 0.5:
            cv2.putText(output, "Prob OUT: " + str(round(outProb * 100, 0)) +"%", (20,140), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(output, "Prob OUT: " + str(outProb * 100) +"%", (20,140), font, 2, (0, 255, 0), 2, cv2.LINE_AA)

    elif vidOp == "Probability Mask":
        output = probMask
        cv2.putText(output, "Max Value: "+str(maxValue), (20,70), font, 2, 255, 2, cv2.LINE_AA)
    
    elif vidOp == "Ball Trajectory":
        output = ballMaskCol

        for i in range(1, frameIndex):
            point1 = trackPredPoints[i]
            point2 = trackPredPoints[i-1]

            currX = point1[0]
            lastX = point2[0]

            if currX != -1 and lastX != -1:
                if i in [i[0] for i in contactFrames]:
                    cv2.line(output, point1[:2], point2[:2], (0,255,0), 2)
                else:
                    cv2.line(output, point1[:2], point2[:2], (0,0,255), 2)
            
        delta = deltaPoints[frameIndex]
        grad = gradPoints[frameIndex]
        rateGrad = rateGradPoints[frameIndex]

        if delta != None:
            delta = round(delta, 4)
        if grad != None:
            grad = round(grad, 4)
        if rateGrad != None:
            rateGrad = round(rateGrad, 4)
        
        cv2.putText(output, "Delta: "+str(delta), (20,70), font, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(output, "Gradient: "+str(grad), (20,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(output, "Gradient Rate: "+str(rateGrad), (20,210), font, 2, (255,255,255), 2, cv2.LINE_AA)
        
    elif vidOp == "Object Detection":
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        output = cv2.add(output, ballMaskCol)
        output = cv2.add(output, lineMaskCol)

        x, y, r = trackPredPoints[frameIndex]
        cv2.putText(output, "Ball Center: ("+str(x)+", "+str(y)+")", (20,70), font, 2, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(output, "Ball Radius: "+str(r), (20,140), font, 2, (0,0,0), 2, cv2.LINE_AA)


    frameIndex += 1
    
    return (output, trackPoints, trackPredPoints, linePoints, gradPoints, rateGradPoints, deltaPoints, frameIndex, contactFrames, contactPrints, outProb)


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