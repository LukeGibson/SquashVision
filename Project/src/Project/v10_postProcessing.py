import cv2
import math
import numpy as np
import numba as nb


def showResult(frame, trackPoints, trackPredPoints, linePoints, gradPoints, rateGradPoints, deltaPoints, frameIndex, contactFrames, contactPrints, outProb, vidOp):
    '''
    Calculates the contact imprint mask at the time of the given frame index.
    Uses contact mask and line mask to calculate the probability the ball was out.
    Displays differnt operated image depening on operation selected by the user.

    :param frame: 
    :param trackPoints: the list of each frames detected ball center and radius
    :param trackPredPoints: the list of each frames predicted ball center and radius
    :param linePoints: the list of each frames line contour object
    :param gradPoints: the list of the gradient of the balls flight in each frame
    :param rateGradPoints: the list of the rate of change of gradient of the balls flight in each frame
    :param deltaPoints: the list of the stright line distance between current and last frames ball centre
    :param frameIndex: the current frame number of playback
    :param contactFrames: the list of pairs of (frame number, radius percentage of contact) where the ball contacts the wall
    :param contactPrints: the list of ball imprint masks
    :param outProb: the probability the ball was out 
    :param vidOp: the output option the user has selected
    :returns: the operated image and updated video data
    '''
    # generate output image base
    height, width = frame.shape[:2]

    ballData = trackPredPoints[frameIndex]
    lineData = linePoints[frameIndex]

    # calculate current frames ball imprint mask if in a contact frame
    if frameIndex in [i[0] for i in contactFrames]:
        ballContact = True
        radiusPercent = contactFrames[[i[0] for i in contactFrames].index(frameIndex)][1]

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
    
    kernel = np.ones((7,7), np.uint8)
    contactMask = cv2.dilate(contactMask, kernel, iterations=2)
    contactMask = cv2.erode(contactMask, kernel, iterations=2)
    contactMaskCol = cv2.dilate(contactMaskCol, kernel, iterations=2)
    contactMaskCol = cv2.erode(contactMaskCol, kernel, iterations=2)

    # create line and ball masks
    lineMask = np.zeros((height, width), np.uint8)
    cv2.drawContours(lineMask, [lineData], -1, 255, 1)
    ballMask = np.zeros((height, width), np.uint8)
    cv2.circle(ballMask, ballData[:2], ballData[2], 255, 1)

    # calculate ball is out probability if in a contact frame
    if frameIndex in [i[0] for i in contactFrames]:
        probMask, newOutProb, maxValue = probOut(contactMask, lineMask)

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

    font = cv2.FONT_HERSHEY_SIMPLEX

    # set output depending on current selected operation
    if vidOp == "Make Decision":
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

        # draw a line between every ball centre up to current frame
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


@nb.jit
def convertLineMask(lineMask):
    '''
    Converts the oiginal line mask so that the pixel value represents the probability the pixel is above the outline.
    Uses numba jit notation for more efficent nested looping of mask pixels.

    :param lineMask: the mask containing just the line contour
    :returns: line mask adapted to hold out area probability
    '''
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
    '''
    Adds probability of detection to the line and contact masks and sum's them together.
    Calculates probability the ball is out by detecting overlap pixel value between line and contact masks.
    Uses ratio of pixels at overlap value to determin finner graularity of probability the ball is out.

    :params contactMask: the imprint mask the ball has made on the wall 
    :params lineMask: the line mask
    :returns: a tuple of the summed probability and line masks, probability the ball is out and max value in the probability mask
    '''
    # convert line mask and add probability pixel values
    lineMask2 = cv2.transpose(lineMask)
    lineMask2 = convertLineMask(lineMask2)
    lineMask2 = cv2.transpose(lineMask2)

    # add contact mask probability pixel values
    kernelSmall = np.ones((3,3), np.uint8)
    contactMaskInner = cv2.erode(contactMask, kernelSmall)
    kernelLarge = np.ones((5,5), np.uint8)
    contactMaskOuter = cv2.dilate(contactMask, kernelLarge)

    contactOuterSize = np.count_nonzero(np.array(contactMaskOuter).flatten() == 255)
    contactSize = np.count_nonzero(np.array(contactMask).flatten() == 255)

    # check inner mask is not dilated too much
    contactInnerSize = np.count_nonzero(np.array(contactMaskInner).flatten() == 255)
    if contactInnerSize < 5:
        contactMaskInner = contactMask
        contactInnerSize = np.count_nonzero(np.array(contactMaskInner).flatten() == 255)

    # convert contact masks values of 255 to 240
    contactMask = cv2.addWeighted(contactMask, (8/17), contactMask, (8/17), 0)
    contactMaskInner = cv2.addWeighted(contactMaskInner, (8/17), contactMaskInner, (8/17), 0)
    contactMaskOuter = cv2.addWeighted(contactMaskOuter, (8/17), contactMaskOuter, (8/17), 0)

    # sum together the component contact mask's and line mask
    contactMask2 = cv2.addWeighted(contactMaskOuter, 0.5, contactMaskInner, 0.5, 0)
    contactMask2 = cv2.addWeighted(contactMask2, (2/3), contactMask, (1/3), 0)

    probMask = cv2.addWeighted(lineMask2, 0.5, contactMask2, 0.5, 0)

    # use max value and number of pixels equal to it in probability mask to set probability out thresholds
    probMaskFlat = np.array(probMask).flatten()
    maxValue = max(probMaskFlat)
    maxCount = np.count_nonzero(probMaskFlat == maxValue)

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

    # calculate probability
    if thresholdB > 0:
        if maxCount > trueContactSize:
            trueContactSize = maxCount

        # the amount probabilty can vary between min and max
        difference = thresholdB - thresholdA

        # calculate percentage of inner pixels that are at this max value
        proprotionMaxContact =  maxCount / trueContactSize

        # probabilty is closer to thresholdB the greater proportion of the pixels were at the max value
        probability = round(thresholdA + (difference * proprotionMaxContact), 2)
    else:
        probability = 0

    return (probMask, probability, maxValue)
