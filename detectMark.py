from enum import Enum
from color import Color
import numpy as np
import math
import cv2
import util

class Mark(Enum):
    TRIANGLE = 3
    RECTANGLE = 4
    HEXAGON = 6
    CIRCLE = 1
    MIX = 0

def detect_circle(img, minR, maxR):
    l = 1
    r = 200
    
    img = cv2.GaussianBlur(img, (5,5), 0)
    while True:
        if cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,10,param1=50,param2=r,minRadius=minR,maxRadius=maxR) is None:
            break
        r += 100

    while r-l > 1:
        mid = (l + r) / 2
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,10,param1=50,param2=mid,minRadius=minR,maxRadius=maxR)
        if circles is None:
            r = mid
        elif circles.shape[1] == 1:
            return circles[0][0]
        else:
            l = mid

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,10,param1=50,param2=l,minRadius=minR,maxRadius=maxR)
    return circles[0][0]

def discriminateColor(just_mark, mask):
    minAve = 100000
    minColor = None
    for color in Color:
        if color is Color.MIX:
            continue
        ave = np.average(np.linalg.norm(just_mark - color.get(), axis = -1) * mask)
        if minAve > ave:
            minAve = ave
            minColor = color

    return minColor

def detectMark(just_center,plt = None):
    lap = cv2.Laplacian(just_center[:,:,1], cv2.CV_64F,ksize=11)
    lap = lap - np.min(lap)
    lap = np.uint8(lap * (255.0/np.max(lap)))
    ret, th = cv2.threshold(lap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # just_center = cv2.cvtColor(just_center, cv2.COLOR_BGR2GRAY)


    circle = detect_circle(lap, 45, 60)

    mark_lu = np.int32(np.maximum(0,circle[:2] - circle[2]))
    mark_rd = np.int32(np.minimum(just_center.shape[0],circle[:2] + circle[2]))


    just_mark = just_center[mark_lu[1]:mark_rd[1],mark_lu[0]:mark_rd[0]]

    mask = np.zeros_like(just_mark)
    cv2.circle(mask, (int(mask.shape[0] / 2),int(mask.shape[0] / 2)), int(mask.shape[0] / 2), (255,255,255), thickness=-1)

    just_mark = np.bitwise_and(just_mark, mask)
    #just_mark = cv2.cvtColor(np.bitwise_and(just_mark, mask),cv2.COLOR_BGR2HSV)
    #just_mark = np.amax(just_mark[:,:,1:3], axis=-1)
    ret, th = cv2.threshold(cv2.cvtColor(just_mark,cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    maxArea = 0
    maxAreaCont = None
    for cont in contours:
        area = cv2.contourArea(cont)
        if maxArea < area:
            maxArea = area
            maxAreaCont = cont

    convex = cv2.convexHull(maxAreaCont, hull=True, clockwise=True)
    convexArc = cv2.arcLength(convex, True)
    arc = cv2.arcLength(maxAreaCont, True)

    if plt:
        buf  = np.zeros_like(th)
        cv2.fillPoly(buf, [maxAreaCont], 1)
        plt.subplot(2,3,4),plt.imshow(buf)
        plt.title('center'), plt.xticks([]), plt.yticks([])
        buf  = np.zeros_like(th)
        cv2.fillPoly(buf, [util.shortcutContour(maxAreaCont, 5)], 1)
        plt.subplot(2,3,5),plt.imshow(buf)
        plt.title('center'), plt.xticks([]), plt.yticks([])
    arcRatio = arc / convexArc

    print(arcRatio)
    if arcRatio > 1.5:
        return Mark.MIX, Color.MIX

    M = cv2.moments(maxAreaCont)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    #cv2.circle(th, (cx,cy), 3, 0)
    # plt.subplot(2,3,5),plt.imshow(th)
    # plt.title('center'), plt.xticks([]), plt.yticks([])

    sz = 480

    dist = np.zeros(sz)

    for ang in range(sz):
        rad = math.radians(ang*(360/sz))
        v = np.array([math.cos(rad), math.sin(rad)])*th.shape[0]*2 + [cx,cy]

        for p1,p2 in zip(maxAreaCont[:,0,:], np.roll(maxAreaCont[:,0,:],1,axis=0)):
            if util.intersect((cx,cy),v, p1,p2) and util.line_intersection_point((cx,cy),v, p1,p2) is not None:
                po = np.array(util.line_intersection_point((cx,cy),v, p1,p2))
                dist[ang] = np.linalg.norm(po - [cx,cy], ord = 2)
                break
        if dist[ang] == 0:
            dist[ang] = dist[ang-1]

    # dist = np.linalg.norm(maxAreaCont - [[cx,cy]], ord = 2, axis = -1)
    #print(dist)
    #dist = dist - np.average(dist)
    freq = np.fft.fft(dist)
    freq2 = np.abs(freq[:(freq.shape[0] // 2)])
    freq2[0] = 0
    #print(freq)
    # buf = np.zeros_like(th)
    # for ang in range(sz):
    #     rad = math.radians(ang*(360/sz))
    #     v = np.array([math.cos(rad), math.sin(rad)])*dist[ang] + [cx,cy]
    #     cv2.circle(buf, tuple(np.int32(v)),1,255)

    buf  = np.zeros_like(th)
    cv2.fillPoly(buf, [maxAreaCont], 1)
    color = discriminateColor(just_mark, buf)

    return Mark(np.argmax(freq2)), color