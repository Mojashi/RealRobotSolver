import cv2
import numpy as np
import math
import util

def detectBoard(image):
    # plt.subplot(2,3,1),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('image'), plt.xticks([]), plt.yticks([])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    blur = cv2.GaussianBlur(gray, (15,15), 0)

    lap = cv2.Laplacian(blur, cv2.CV_64F,ksize=11)
    print(lap.shape)

    lap = lap - np.min(lap)
    lap = np.uint8(lap * (255.0/np.max(lap)))

    # plt.subplot(2,3,2),plt.imshow(lap,cmap = 'gray')
    # plt.title('laplacian k=11'), plt.xticks([]), plt.yticks([])
    ret, th = cv2.threshold(lap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    closing = np.uint8(cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel))

    # plt.subplot(2,3,2),plt.imshow(closing,cmap = 'gray')
    # plt.title('binary'), plt.xticks([]), plt.yticks([])

    shortEdge = min(th.shape[0],th.shape[0])
    maxLinegap = shortEdge*0.05
    lines = cv2.HoughLinesP(closing, 1, np.pi / 720, 80,minLineLength=shortEdge*0.7, maxLineGap=maxLinegap)


    arrive = np.ones(len(lines))

    erl = [[] for i in range(len(lines))]

    for i in range(len(lines)):
        if arrive[i] == 0:
            continue
        p1 = (lines[i][0][0],lines[i][0][1])
        p2 = (lines[i][0][2],lines[i][0][3])

        for j in range(len(lines)):
            if i== j:
                continue
            p3 = (lines[j][0][0],lines[j][0][1])
            p4 = (lines[j][0][2],lines[j][0][3])
            if arrive[j] == 0 or util.lenline(p1,p2) < util.lenline(p3,p4):
                continue
            if math.acos(min(1,abs(util.coslines(p1,p2,p3,p4)))) < (math.pi / 18) and util.distanceL2L(p1,p2,p3,p4) < shortEdge * 0.03:
                arrive[j] = 0
                erl[j].append(i)

    print(erl)
    oklines = []

    for i in range(len(lines)):
        if arrive[i] == 0:
            continue
        oklines.append(lines[i])

    endCount = np.zeros(len(oklines))
    frl = [[] for i in range(len(oklines))]

    for i in range(len(oklines)):
        for j in range(len(oklines)):
            p1 = (oklines[i][0][0],oklines[i][0][1])
            p2 = (oklines[i][0][2],oklines[i][0][3])
            p3 = (oklines[j][0][0],oklines[j][0][1])
            p4 = (oklines[j][0][2],oklines[j][0][3])

            if i == j or math.acos(min(1,abs(util.coslines(p1,p2,p3,p4)))) < (math.pi / 6):
                continue
            if util.distanceV2L(p1, p3, p4) < shortEdge * 0.03:
                endCount[j]+=1
                frl[j].append(i)
            if util.distanceV2L(p2, p3, p4) < shortEdge * 0.03:
                endCount[j]+=1
                frl[j].append(i)



    i = 0
    endCount, oklines,frl =zip( *sorted(list(zip(endCount, oklines,frl)),key=lambda x:-x[0]))

    print(len(lines))
    print(len(oklines))
    print(list(zip(endCount,frl)))
    if len(oklines) < 4:
        print("not enough lines")
        exit()

    lineimg = np.copy(image)
    for line in oklines:
        #cv2.imwrite(str(i)+".jpg",cv2.line(np.copy(image), (line[0][0],line[0][1]),(line[0][2],line[0][3]), (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)), 3))
        color = (0,255,0)
        if i < 4:
            color = (0,0,255)
        lineimg = cv2.line(lineimg, (line[0][0],line[0][1]),(line[0][2],line[0][3]),color , 3)
        i+=1


    # plt.subplot(2,3,3),plt.imshow(cv2.cvtColor(lineimg,cv2.COLOR_BGR2RGB),cmap = 'gray')
    # plt.title('lines'), plt.xticks([]), plt.yticks([])

    points = []

    for i in range(4):
        for j in range(i + 1, 4):
            p1 = (oklines[i][0][0],oklines[i][0][1])
            p2 = (oklines[i][0][2],oklines[i][0][3])
            p3 = (oklines[j][0][0],oklines[j][0][1])
            p4 = (oklines[j][0][2],oklines[j][0][3])
            if math.acos(min(1,abs(util.coslines(p1,p2,p3,p4)))) > (math.pi / 6):
                points.append(util.line_intersection_point(p1,p2,p3,p4))

    if len(points) != 4:
        print("invalid")
        exit()

    return np.float64(points)