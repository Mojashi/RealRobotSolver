from color import Color
import cv2
from color import cvtLab
import numpy as np

def extractColor(img, color, k):
    img = cvtLab(img)
    color = cvtLab(color)
    
    dist = np.linalg.norm(img - color,axis=-1)
    border = np.sort(np.reshape(dist, -1))[k] 
    mask = np.uint8(dist < border)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    maxScore = 0
    maxScoreLabel = 0
    maxDist = np.max(dist)
    dist = maxDist - dist
    for i in range(1,retval):
        score = np.sum(dist * (labels == (i + 1)))
        if maxScore < score:
            maxScore = score
            maxScoreLabel = i + 1
    
    return centroids[maxScoreLabel], np.uint8(maxScoreLabel == labels)


def detectRobot(just_board):
    robotPos = [[],[],[],[]]
    robotMask = [None,None,None,None]

    robotPos[0],robotMask[0] = extractColor(just_board, Color.BLUE.get(), 2000)
    robotPos[1],robotMask[1] = extractColor(just_board, Color.RED.get(), 2000)
    robotPos[2],robotMask[2] = extractColor(just_board, Color.GREEN.get(), 2000)
    robotPos[3],robotMask[3] = extractColor(just_board, Color.YELLOW.get(), 2000)

    gridSz = just_board.shape[0] / 16

    for i in range(4):
        robotPos[i] = np.int32(robotPos[i] / gridSz)
        cv2.circle(just_board, tuple(np.int32(robotPos[i]*gridSz+ gridSz/2)), 5, (255,255,255),-1)

    return robotPos