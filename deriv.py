import numpy as np
import cv2
import tkinter
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

def approxRectangle(contour):
    arclen = cv2.arcLength(contour,True)
    l = 0.0
    r = 99999.0

    poly = cv2.approxPolyDP(cv2.convexHull(contour), 0,True)
    if len(poly) < 4:
        return poly

    poly = cv2.approxPolyDP(cv2.convexHull(contour), 99999*arclen,True)
    if len(poly) < 4:
        return poly


    while True:
        mid = (l + r) / 2
        poly = cv2.approxPolyDP(cv2.convexHull(contour), arclen * mid,True)
        if len(poly) == 4:
            return poly
        elif len(poly) < 4:
            r = mid
        else:
            l = mid

image = cv2.imread('board5.jpg')

rszp = (np.sqrt((image.shape[0] * image.shape[1]) / 300000))
print(rszp)
image = cv2.resize(image, (int(image.shape[1]/rszp),int(image.shape[0] /rszp)))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blur = cv2.GaussianBlur(gray, (15,15), 0)

lap = cv2.Laplacian(blur, cv2.CV_64F,ksize=11)
print(lap.shape)

lap = lap - np.min(lap)
lap = np.uint8(lap * (255.0/np.max(lap)))

plt.subplot(1,3,1),plt.imshow(lap,cmap = 'gray')
plt.title('Laplacian k=31'), plt.xticks([]), plt.yticks([])
ret, th = cv2.threshold(lap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, th = cv2.threshold(lap, ret + 50, 255, cv2.THRESH_BINARY)

plt.subplot(1,3,2),plt.imshow(th,cmap = 'gray')
plt.title('binary'), plt.xticks([]), plt.yticks([])

kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

plt.subplot(1,3,3),plt.imshow(closing,cmap = 'gray')
plt.title('binary'), plt.xticks([]), plt.yticks([])

contours, hierarchy = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

boardcnt = []
maxlen = 0

for cnt in contours:
    poly = approxRectangle(cnt)
    arc = cv2.arcLength(poly,True)
    if len(poly) == 4 and arc > maxlen:
        maxle = arc
        boardcnt = [poly]

#scont = [ (menseki(cv2.minAreaRect(cnt)) ,cnt) for cnt in contours for menseki in [lambda x:x.width*x.height]]
#scont.sort(key=lambda x:-x[0])
#contours = np.array([cnt[1] for cnt in scont])
print(len(contours))


cntimg = cv2.drawContours(image, boardcnt, -1, (0,255,0), 2)

plt.figure()
plt.imshow(cv2.cvtColor(cntimg,cv2.COLOR_BGR2RGB))


plt.show()