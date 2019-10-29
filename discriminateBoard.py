import cv2
import numpy as np

def discriminateBoard(just_board):
    w,h,_ = just_board.shape
    quater = [  cv2.cvtColor(np.rot90(np.rot90(just_board[0:250,0:250])), cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(np.rot90(just_board[250:500,0:250]), cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(just_board[250:500,250:500], cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(np.rot90(np.rot90(np.rot90(just_board[0:250,250:500]))), cv2.COLOR_BGR2GRAY)]


    quater_rank = [[],[],[],[]]
    temps = []

    for i in range(17):
        template = cv2.cvtColor(cv2.imread('ricochetboard/' + str(i + 1) + ".jpg"), cv2.COLOR_BGR2GRAY)
        template = cv2.resize(template, dsize=(250, 250))

        for j in range(4):
            tmpmatch = cv2.matchTemplate(quater[j], template, cv2.TM_CCOEFF_NORMED)
            min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(tmpmatch)
            quater_rank[j].append((-max_value, max_pt, i))

    ret = []
    for i in range(4):
        quater_rank[i].sort()
        ret.append(quater_rank[i][0][2] + 1)

    return ret