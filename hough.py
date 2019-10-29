import numpy as np
import random
import cv2
import tkinter
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import math

from detectBoard import detectBoard
from detectMark import Mark
from detectMark import detectMark
from detectRobot import detectRobot
from discriminateBoard import discriminateBoard


from util import padding

image = cv2.imread('testcase/board9.jpg')
original_image = image
rszp = (np.sqrt((image.shape[0] * image.shape[1]) / 300000))
image = cv2.resize(image, (int(image.shape[1]/rszp),int(image.shape[0] /rszp)))

plt.subplot(2,3,1),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('image'), plt.xticks([]), plt.yticks([])

boardSz = 500
centerSz = [int(boardSz*4*840/2000),int(boardSz*4-boardSz*4*840/2000)]
boardPoints = list(detectBoard(image)*rszp)

center = [0,0]
for i in range(4):
    center[0] += boardPoints[i][0] / 4.0
    center[1] += boardPoints[i][1] / 4.0

boardPoints.sort(key = lambda x:-math.atan2(x[0] - center[1], x[1] - center[0]))
boardPoints = np.float32(boardPoints)
just_board_pts = [[boardSz,0],[boardSz,boardSz],[0,boardSz],[0,0]]
M = cv2.getPerspectiveTransform(boardPoints,np.float32(just_board_pts))
just_board =  cv2.warpPerspective(original_image,M,(boardSz,boardSz))

plt.subplot(2,3,2),plt.imshow(cv2.cvtColor(just_board, cv2.COLOR_BGR2RGB))
plt.title('board'), plt.xticks([]), plt.yticks([])

just_center_pts = [[boardSz*4,0],[boardSz*4,boardSz*4],[0,boardSz*4],[0,0]]
M = cv2.getPerspectiveTransform(boardPoints,np.float32(just_center_pts))
just_center = cv2.warpPerspective(original_image,M,(boardSz*4,boardSz*4))[centerSz[0]:centerSz[1],centerSz[0]:centerSz[1]]
just_center = padding(just_center, 20, 255)

plt.subplot(2,3,3),plt.imshow(cv2.cvtColor(just_center, cv2.COLOR_BGR2RGB))
plt.title('center'), plt.xticks([]), plt.yticks([])

boardNum = discriminateBoard(just_board)
print(boardNum)

mark = detectMark(just_center,plt)
print(mark)

robots = detectRobot(just_board)
print(robots)

plt.show()