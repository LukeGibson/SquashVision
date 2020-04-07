import cv2
import numpy as np
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