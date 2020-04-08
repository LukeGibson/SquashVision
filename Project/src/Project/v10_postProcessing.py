import cv2
import math
import numpy as np
import numba as nb


def showResult(frame, trackPoints, trackPredPoints, linePoints, gradPoints, rateGradPoints, deltaPoints, frameIndex, contactFrames, contactPrints, outProb, vidOp):
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
        probMask, newOutProb, maxValue = probOut(contactMask, lineMask)

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
        contactMaskInner = contactMask
        contactInnerSize = np.count_nonzero(np.array(contactMaskInner).flatten() == 255)

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

    # only calculate the probabilty if thesholdB > 0
    if thresholdB > 0:

        # error check
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
