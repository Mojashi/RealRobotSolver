import cv2
from matplotlib import pyplot as plt
import numpy as np


def thining(_img):
    img = _img.copy()
    neigh = [(0,-1),(1,-1),(1,0),(1,1), (0,1), (-1, 1), (-1, 0), (-1,-1)]
    w,h = img.shape

    def get(x,y,p):
        return img[x + neigh[p][0]][y + neigh[p][1]]

    def N(x,y):
        cou = 0
        for nei in neigh:
            cou += int(img[x + nei[0]][y + nei[1]] == 0)
        return cou
        
    def S(x,y):
        bef = img[x + neigh[7][0]][y + neigh[7][1]]
        cou = 0
        for nei in neigh:
            cou += int(img[x + nei[0]][y + nei[1]] != bef)
            bef = img[x + nei[0]][y + nei[1]]
        return
    
    while(True):
        changed = False
        for x in range(1, w - 1):
            for y in range(1, h - 1):
                if img[x][y] == 255:
                    continue

                nv = N(x,y)
                sv = S(x,y)
                if 2 <= nv and nv <= 6 and sv == 1 and \
                    get(x,y,2)*get(x,y,4)*get(x,y,6) == 0 and get(x,y,4)*get(x,y,6)*get(x,y,8) == 0:
                    img[x][y] = 0
                    
                if 2 <= nv and nv <= 6 and sv == 1 and \
                    get(x,y,2)*get(x,y,4)*get(x,y,6) == 0 and get(x,y,2)*get(x,y,6)*get(x,y,8) == 0:
                    img[x][y] = 0

    return img

board = cv2.imread("ricochetboard/1.jpg")
gboard  =cv2.cvtColor(board,cv2.COLOR_BGR2GRAY)
pboard = [[max(b[1],b[2]) for b in a] for a in cv2.cvtColor(board,cv2.COLOR_BGR2HSV_FULL)]
#print(pboard)

bboard = cv2.adaptiveThreshold(np.uint8(pboard),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,201,20)
#_,bboard = cv2.threshold(np.uint8(pboard),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.figure()
plt.imshow(bboard, cmap="gray")


neiborhood = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],
             np.uint8)
#膨張
board_erode = cv2.erode(bboard,neiborhood,iterations=10)
#収縮
board_dilate = cv2.dilate(board_erode,neiborhood,iterations=10)


plt.figure()
plt.imshow(board_dilate, cmap="gray")
plt.show()

cv2.imwrite("res.jpg", board_dilate)