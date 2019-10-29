import cv2
import numpy as np


def padding(image, pad = 1, v = 1):
    shape = list(image.shape)
    shape[0] += pad*2
    shape[1] += pad*2
    print(shape)
    result = np.full(tuple(shape), v, dtype=image.dtype)
    result[pad:-pad, pad:-pad] = image
    return result

def unpadding(image):
    return image[1:-1, 1:-1]
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

# def shortcutContour(cnt, k):
#     dist = np.ndarray(cnt.shape[0])
#     dist[0] = 0
#     shu = 0

#     for i, (fr, to) in enumerate(zip(cnt[:,0], np.roll(cnt[:,0], -1))):
#         if i == cnt.shape[0]-1:
#             shu = lenline(fr,to) + dist[i]
#             break
#         dist[i + 1] = lenline(fr,to) + dist[i]
    
#     shortcut = []

#     i = 0

#     while int(i) < int(cnt.shape[0]):
#         maxDist = lenline(cnt[i][0], cnt[(i + 1)%cnt.shape[0]][0])
#         maxPoint = cnt[(i + 1)%cnt.shape[0]]

#         for j in range(cnt.shape[0]):
#             dis = np.linalg.norm(cnt[i][0] - cnt[j][0])
#             if dis <= k:
#                 shudist = min(abs(dist[i] - dist[j]),abs(dist[i] - shu - dist[j]))
#                 if shudist > maxDist:
#                     maxDist = shudist
#                     maxPoint = j

#         i = maxPoint
#         shortcut.append(cnt[maxPoint])
#         i+=1
    
#     return np.array(shortcut)

                
def shortcutContour(cnt, k):
    hull = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    cutPoint = np.zeros(cnt.shape[0])

    dist = np.ndarray(cnt.shape[0])
    dist[0] = 0
    shu = 0

    for i, (fr, to) in enumerate(zip(cnt[:,0], np.roll(cnt[:,0], -1))):
        if i == cnt.shape[0]-1:
            shu = lenline(fr,to) + dist[i]
            break
        dist[i + 1] = lenline(fr,to) + dist[i]
    
    for defect in defects:
        s,e,f,d = defect[0]

        poss = np.where(np.linalg.norm(cnt - cnt[f]) < k)
        for pos in poss:
            shudist = min(abs(dist[f] - dist[pos]),abs(dist[f] - shu - dist[pos]),abs(dist[f] + shu - dist[pos]))
            