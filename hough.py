import numpy as np
import random
import cv2
from scipy import fftpack
import tkinter
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import math
# 与えられた引数全てについての論理積を返すメソッドです。
def multi_logical_and(*args):
    result = np.copy(args[0])
    for arg in args:
        result = np.logical_and(result, arg)
    return result

# 2値画像について、周囲1ピクセルをFalseで埋めるメソッドです
def padding(image, pad = 1, v = 1):
    shape = list(image.shape)
    shape[0] += pad*2
    shape[1] += pad*2
    print(shape)
    result = np.full(tuple(shape), v, dtype=image.dtype)
    result[pad:-pad, pad:-pad] = image
    return result

# paddingの逆です
def unpadding(image):
    return image[1:-1, 1:-1]

# そのピクセルの周囲のピクセルの情報を格納したarrayを返します。
def generate_mask(image):
    row, col = np.shape(image)
    p2 = np.zeros((row, col)).astype(bool)
    p3 = np.zeros((row, col)).astype(bool)
    p4 = np.zeros((row, col)).astype(bool)
    p5 = np.zeros((row, col)).astype(bool)
    p6 = np.zeros((row, col)).astype(bool)
    p7 = np.zeros((row, col)).astype(bool)
    p8 = np.zeros((row, col)).astype(bool)
    p9 = np.zeros((row, col)).astype(bool)
    #上
    p2[1:row-1, 1:col-1] = image[0:row-2, 1:col-1]
    #右上
    p3[1:row-1, 1:col-1] = image[0:row-2, 2:col]
    #右
    p4[1:row-1, 1:col-1] = image[1:row-1, 2:col]
    #右下
    p5[1:row-1, 1:col-1] = image[2:row, 2:col]
    #下
    p6[1:row-1, 1:col-1] = image[2:row, 1:col-1]
    #左下
    p7[1:row-1, 1:col-1] = image[2:row, 0:col-2]
    #左
    p8[1:row-1, 1:col-1] = image[1:row-1, 0:col-2]
    #左上
    p9[1:row-1, 1:col-1] = image[0:row-2, 0:col-2]
    return (p2, p3, p4, p5, p6, p7, p8, p9)

# 周囲のピクセルを順番に並べたときに白→黒がちょうど1箇所だけあるかどうかを判定するメソッドです。
def is_once_change(p_tuple):
    number_change = np.zeros_like(p_tuple[0])
    # P2~P9,P2について、隣接する要素の排他的論理和を取った場合のTrueの個数を数えます。
    for i in range(len(p_tuple) - 1):
        number_change = np.add(number_change, np.logical_xor(p_tuple[i], p_tuple[i+1]).astype(int))
    number_change = np.add(number_change, np.logical_xor(p_tuple[7], p_tuple[0]).astype(int))
    array_two = np.ones_like(p_tuple[0]) * 2

    return np.equal(number_change, array_two)

# 周囲の黒ピクセルの数を数え、2以上6以下となっているかを判定するメソッドです。
def is_black_pixels_appropriate(p_tuple):
    number_of_black_pxels = np.zeros_like(p_tuple[0])
    array_two = np.ones_like(p_tuple[0]) * 2
    array_six = np.ones_like(p_tuple[0]) * 6
    for p in p_tuple:
        number_of_black_pxels = np.add(number_of_black_pxels, p.astype(int))
    greater_two = np.greater_equal(number_of_black_pxels, array_two)
    less_six = np.less_equal(number_of_black_pxels, array_six)
    return np.logical_and(greater_two, less_six)

def step1(image, p_tuple):
    #条件1
    condition1 = np.copy(image)

    #条件2
    condition2 = is_once_change(p_tuple)

    #条件3
    condition3 = is_black_pixels_appropriate(p_tuple)

    #条件4
    condition4 = np.logical_not(multi_logical_and(p_tuple[0], p_tuple[2], p_tuple[4]))

    #条件5
    condition5 = np.logical_not(multi_logical_and(p_tuple[2], p_tuple[4], p_tuple[6]))

    return np.logical_xor(multi_logical_and(condition1, condition2, condition3, condition4, condition5), image)

def step2(image, p_tuple):
    #条件1
    condition1 = np.copy(image)

    #条件2
    condition2 = is_once_change(p_tuple)

    #条件3
    condition3 = is_black_pixels_appropriate(p_tuple)

    #条件4
    condition4 = np.logical_not(np.logical_and(p_tuple[0], np.logical_and(p_tuple[2], p_tuple[6])))

    #条件5
    condition5 = np.logical_not(np.logical_and(p_tuple[0], np.logical_and(p_tuple[4], p_tuple[6])))

    return np.logical_xor(multi_logical_and(condition1, condition2, condition3, condition4, condition5), image)

# 2値化画像を細線化して返すメソッドです。
def ZhangSuen(image):

    image = padding(image)

    while True:
        old_image = np.copy(image)

        p_tuple = generate_mask(image)
        image = step1(image, p_tuple)
        p_tuple = generate_mask(image)        
        image = step2(image, p_tuple)

        if (np.array_equal(old_image, image)):
            break

    return unpadding(image)
def distanceV2L(point, p1,p2):
    x0,y0 = point
    x1,y1 = p1
    x2,y2 = p2

    if  abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) == 0:
        return 0

    return abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / np.sqrt(np.square(x2-x1) + np.square(y2-y1))

def distanceL2L(p1,p2,p3,p4):
    if intersect(p1,p2,p3,p4):
        return 0
    return min(distanceV2L(p1,p3,p4),distanceV2L(p2,p3,p4),distanceV2L(p3,p1,p2),distanceV2L(p4,p1,p2))

def calcRadian(line):
    return math.atan2(line[0][1] - line[0][0], line[0][3] - line[0][2])

def distanceRadian(a1,a2):
    return 

def line_intersection_point(p1,p2,p3,p4):
    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*(p1,p2)), det(*(p3,p4)))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y



def intersect(p1, p2, p3, p4):
    tc1 = int((p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0]))
    tc2 = int((p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0]))
    td1 = int((p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0]))
    td2 = int((p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0]))
    d= tc1*tc2<=0 and td1*td2<=0
    return d
def lenline(p1,p2):
    return np.sqrt((p2[0] - p1[0])*(p2[0] - p1[0])+(p2[1] - p1[1])*(p2[1] - p1[1]))

def coslines(p1,p2,p3,p4):
    d=((p2[0] - p1[0])*(p4[0] - p3[0]) + (p2[1] - p1[1]) * (p4[1] - p3[1])) / (lenline(p1,p2)*lenline(p3,p4))
    return d

#-*- coding:utf-8 -*-
import cv2
import numpy as np

def template_matching_zncc(src, temp):
    # 画像の高さ・幅を取得
    h, w = src.shape
    ht, wt = temp.shape
   
    # スコア格納用の2次元リスト
    score = np.empty((h-ht, w-wt))

    # 配列のデータ型をuint8からfloatに変換
    src = np.array(src, dtype="float")
    temp = np.array(temp, dtype="float")

    # テンプレート画像の平均画素値
    mu_t = np.mean(temp)

    # 走査
    for dy in range(0, h - ht):
        for dx in range(0, w - wt):
            # 窓画像
            roi = src[dy:dy + ht, dx:dx + wt]
            # 窓画像の平均画素値
            mu_r = np.mean(roi)
            # 窓画像 - 窓画像の平均
            roi = roi - mu_r
            # テンプレート画像 - 窓画像の平均
            temp = temp - mu_t

            # ZNCCの計算式
            num = np.sum(roi * temp)
            den = np.sqrt( np.sum(roi ** 2) ) * np.sqrt( np.sum(temp ** 2) ) 
            if den == 0: score[dy, dx] = 0
            score[dy, dx] = num / den

    # スコアが最大(1に最も近い)の走査位置を返す
    pt = np.unravel_index(score.argmin(), score.shape)

    return (pt[1], pt[0])

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

image = cv2.imread('testcase/board3.jpg')
original_image = image

rszp = (np.sqrt((image.shape[0] * image.shape[1]) / 300000))
print(rszp)
image = cv2.resize(image, (int(image.shape[1]/rszp),int(image.shape[0] /rszp)))

plt.subplot(2,3,1),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('image'), plt.xticks([]), plt.yticks([])

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

plt.subplot(2,3,2),plt.imshow(closing,cmap = 'gray')
plt.title('binary'), plt.xticks([]), plt.yticks([])

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
        if arrive[j] == 0 or lenline(p1,p2) < lenline(p3,p4):
            continue
        if math.acos(min(1,abs(coslines(p1,p2,p3,p4)))) < (math.pi / 18) and distanceL2L(p1,p2,p3,p4) < shortEdge * 0.03:
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

        if i == j or math.acos(min(1,abs(coslines(p1,p2,p3,p4)))) < (math.pi / 6):
            continue
        if distanceV2L(p1, p3, p4) < shortEdge * 0.03:
            endCount[j]+=1
            frl[j].append(i)
        if distanceV2L(p2, p3, p4) < shortEdge * 0.03:
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


plt.subplot(2,3,3),plt.imshow(cv2.cvtColor(lineimg,cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('lines'), plt.xticks([]), plt.yticks([])

points = []

for i in range(4):
    for j in range(i + 1, 4):
        p1 = (oklines[i][0][0],oklines[i][0][1])
        p2 = (oklines[i][0][2],oklines[i][0][3])
        p3 = (oklines[j][0][0],oklines[j][0][1])
        p4 = (oklines[j][0][2],oklines[j][0][3])
        if math.acos(min(1,abs(coslines(p1,p2,p3,p4)))) > (math.pi / 6):
            points.append(line_intersection_point(p1,p2,p3,p4))

if len(points) != 4:
    print("invalid")
    plt.show()
    exit()
    
just_board_pts = [[500,0],[500,500],[0,500],[0,0]]

center = [0,0]
for i in range(4):
    center[0] += points[i][0] / 4.0
    center[1] += points[i][1] / 4.0

points.sort(key = lambda x:-math.atan2(x[0] - center[1], x[1] - center[0]))


M = cv2.getPerspectiveTransform(np.float32(points)*rszp,np.float32(just_board_pts))
just_board = cv2.warpPerspective(original_image,M,(500,500))


just_center_pts = [[2000,0],[2000,2000],[0,2000],[0,0]]
M = cv2.getPerspectiveTransform(np.float32(points)*rszp,np.float32(just_center_pts))
just_center = cv2.warpPerspective(original_image,M,(2000,2000))[840:1160, 840:1160]

plt.subplot(2,3,4),plt.imshow(cv2.cvtColor(just_board,cv2.COLOR_BGR2RGB))
plt.title('board'), plt.xticks([]), plt.yticks([])

quater = [  cv2.cvtColor(np.rot90(np.rot90(just_board[0:250,0:250])), cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(np.rot90(just_board[250:500,0:250]), cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(just_board[250:500,250:500], cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(np.rot90(np.rot90(np.rot90(just_board[0:250,250:500]))), cv2.COLOR_BGR2GRAY)]


just_center = padding(just_center, 20, 255)

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

just_mark = cv2.cvtColor(np.bitwise_and(just_mark, mask),cv2.COLOR_BGR2GRAY)
#just_mark = cv2.cvtColor(np.bitwise_and(just_mark, mask),cv2.COLOR_BGR2HSV)
#just_mark = np.amax(just_mark[:,:,1:3], axis=-1)
ret, th = cv2.threshold(just_mark, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

arcRatio = arc / convexArc

print(arcRatio)
if arcRatio > 1.3:
    print()

M = cv2.moments(maxAreaCont)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

cv2.circle(th, (cx,cy), 3, 0)
# plt.subplot(2,3,5),plt.imshow(th)
# plt.title('center'), plt.xticks([]), plt.yticks([])

sz = 480

dist = np.zeros(sz)

for ang in range(sz):
    rad = math.radians(ang*(360/sz))
    v = np.array([math.cos(rad), math.sin(rad)])*th.shape[0]*2 + [cx,cy]

    for p1,p2 in zip(maxAreaCont[:,0,:], np.roll(maxAreaCont[:,0,:],1,axis=0)):
        if intersect((cx,cy),v, p1,p2) and line_intersection_point((cx,cy),v, p1,p2) is not None:
            po = np.array(line_intersection_point((cx,cy),v, p1,p2))
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
buf = np.zeros_like(th)
for ang in range(sz):
    rad = math.radians(ang*(360/sz))
    v = np.array([math.cos(rad), math.sin(rad)])*dist[ang] + [cx,cy]
    cv2.circle(buf, tuple(np.int32(v)),1,255)

print(str(np.argmax(freq2)) + "角形")
#print(freq2)
#freq[7:] = 0
plt.subplot(2,3,5),plt.imshow(th)
# plt.subplot(2,3,6),plt.plot(freq2)

# red = np.array([40,20,233])
# green = np.array([40,180,40])
# yellow = np.array([60,170,250])
# blue = np.array([100,0,0])
# white = np.array([255,255,255])

red = cv2.cvtColor(np.uint8([[[0,0,255]]]), cv2.COLOR_BGR2Lab)
green = cv2.cvtColor(np.uint8([[[0,255,0]]]), cv2.COLOR_BGR2Lab)
yellow = cv2.cvtColor(np.uint8([[[0,255,255]]]), cv2.COLOR_BGR2Lab)
blue = cv2.cvtColor(np.uint8([[[255,0,0]]]), cv2.COLOR_BGR2Lab)
labimg = cv2.cvtColor(just_board, cv2.COLOR_BGR2Lab)

red = np.float32(red)
green = np.float32(green)
labimg = np.float32(labimg)
yellow = np.float32(yellow)
red[:,:,0]  *=100/255
red[:,:,1]  -= 128
red[:,:,2]  -= 128

yellow[:,:,0] *=100/255
yellow[:,:,1] -= 128
yellow[:,:,2] -= 128
green[:,:,0]  *=100/255
green[:,:,1]  -= 128
green[:,:,2]  -= 128
labimg[:,:,0]  *=100/255
labimg[:,:,1]  -= 128
labimg[:,:,2]  -= 128

#ret, bdth = cv2.threshold(cv2.cvtColor(just_board, cv2.COLOR_BGR2HSV)[:,:,1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# ret, bdth = cv2.threshold(cv2.cvtColor(just_board, cv2.COLOR_BGR2HSV)[:,:,1], ret + 50, 255, cv2.THRESH_BINARY)
def extractColor(img, col, k):
    rdot = np.linalg.norm(img - col,axis=-1) #- cv2.cvtColor(img, cv2.COLOR_Lab2HSV)[:,:,1]**(1/2)
    border = np.sort(np.reshape(rdot, -1))[k] 
    return (rdot < border)

# buf = np.minimum(np.reshape(extractColor(just_board, yellow,2000),(500,500,1))*np.tile(yellow,(500,500,1))+
#     np.reshape(extractColor(just_board, green,2000),(500,500,1))*np.tile(green,(500,500,1))+
#     np.reshape(extractColor(just_board, blue,2000),(500,500,1))*np.tile(blue,(500,500,1))+
#     np.reshape(extractColor(just_board, red,2000),(500,500,1))*np.tile(red,(500,500,1)), 255)

norm = np.linalg.norm(labimg - red,axis=-1)
plt.subplot(2,3,6),plt.imshow(norm / np.max(norm) * 255)#extractColor(cv2.cvtColor(just_board, cv2.COLOR_BGR2Lab), yellow,2000))

# convex = cv2.convexHull(maxAreaCont, clockwise=True)
# #approx = cv2.approxPolyDP(convex,0.02*cv2.arcLength(convex,True) ,True)
# buf =np.zeros_like(th)
# cv2.fillPoly(buf,[convex],255)

# bef = convex[-1][0]
# minRad = 333333
# minPoint = None
# for cur, nex in zip(convex[:,0,:], np.roll(convex, -1)[:,0,:]):
#     v1 = bef - cur
#     v2 = nex - cur
#     cs = np.dot(v1,v2)/np.linalg.norm(v1, ord=2)/np.linalg.norm(v2, ord=2)
#     rad = abs(math.acos(cs))
#     if minRad > cs:
#         minRad = cs
#         minPoint = cur
#     bef = cur

# cv2.circle(buf, tuple(minPoint), 3, 255)
# print(minPoint)
# print(minRad)
# plt.subplot(2,3,6),plt.imshow(buf)
# plt.title('center'), plt.xticks([]), plt.yticks([])

quater_rank = [[],[],[],[]]
temps = []

for i in range(17):
    template = cv2.cvtColor(cv2.imread('ricochetboard/' + str(i + 1) + ".jpg"), cv2.COLOR_BGR2GRAY)
    template = cv2.resize(template, dsize=(250, 250))

    for j in range(4):
        tmpmatch = cv2.matchTemplate(quater[j], template, cv2.TM_CCOEFF_NORMED)
        min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(tmpmatch)
        quater_rank[j].append((-max_value, max_pt, i))

for i in range(4):
    quater_rank[i].sort()
    print(quater_rank[i][0][2] + 1)


# template = cv2.cvtColor(cv2.imread("simbol/scan-003.jpg"),cv2.COLOR_BGR2GRAY)
# template=cv2.resize(template, dsize=(250,250))
# tmpmatch = cv2.matchTemplate(cv2.cvtColor(just_board,cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
# min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(tmpmatch)
# pt = max_pt
# print(pt)
# h, w = template.shape

# # テンプレートマッチングの結果を出力
# just_board = cv2.rectangle(just_board, (pt[0], pt[1] ), (pt[0] + w, pt[1] + h), (0,200,0), 3)





# template_path = "simbol/"
# template_filename = "scan-018.jpg"

# akaze = cv2.AKAZE_create() 

# # 文字画像を読み込んで特徴量計算
# expand_template=0.4
# whitespace = 20
# template_temp = cv2.imread(template_path + template_filename, 0)
# height, width = template_temp.shape[:2]
# template_img=np.ones((height+whitespace*2, width+whitespace*2),np.uint8)*255
# template_img[whitespace:whitespace + height, whitespace:whitespace+width] = template_temp
# template_img = cv2.resize(template_img, None, fx = expand_template, fy = expand_template)
# kp_temp, des_temp = akaze.detectAndCompute(template_img, None)

# # 間取り図を読み込んで特徴量計算
# expand_sample = 1
# sample_img = cv2.cvtColor(just_center, cv2.COLOR_BGR2GRAY)
# sample_img = cv2.resize(sample_img, None, fx = expand_sample, fy = expand_sample)
# kp_samp, des_samp = akaze.detectAndCompute(sample_img, None)

# # 特徴量マッチング実行
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des_temp, des_samp, k=2)

# # マッチング精度が高いもののみ抽出
# ratio = 0.965
# good = []
# for m, n in matches:
#     if m.distance < ratio * n.distance:
#         good.append(m)

# if len(good)>5:
#     src_pts = np.float32([ kp_temp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp_samp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()

#     h,w = template_img.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
#     print(dst)
#     print(pts)
#     wp = cv2.warpPerspective(sample_img,cv2.getPerspectiveTransform(dst, pts),(h,w))

#     plt.subplot(2,3,6),plt.imshow(cv2.cvtColor(wp,cv2.COLOR_BGR2RGB))
#     plt.title('lines'), plt.xticks([]), plt.yticks([])


#     sample_img = cv2.polylines(sample_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

# else:
#     print("Not enough matches are found - %d/%d" % (len(good), 5))
#     matchesMask = None

# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)

# img3 = cv2.drawMatches(template_img, kp_temp,sample_img,kp_samp,good,None,**draw_params)


plt.show()