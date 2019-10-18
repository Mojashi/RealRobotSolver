import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像読み込み先のパス，結果保存用のパスの設定
template_path = "simbol/"
template_filename = "4wallwb.jpg"

sample_path = ""
sample_filename = "ricochetboard/4.jpg"

result_path = "result/"
result_name = "akaze_res.jpg"

akaze = cv2.AKAZE_create() 

# 文字画像を読み込んで特徴量計算
expand_template=0.5
whitespace = 20
template_temp = cv2.imread(template_path + template_filename, 0)
height, width = template_temp.shape[:2]
template_img=np.ones((height+whitespace*2, width+whitespace*2),np.uint8)*255
template_img[whitespace:whitespace + height, whitespace:whitespace+width] = template_temp
template_img = cv2.resize(template_img, None, fx = expand_template, fy = expand_template)
kp_temp, des_temp = akaze.detectAndCompute(template_img, None)

# 間取り図を読み込んで特徴量計算
expand_sample = 1
sample_img = cv2.imread(sample_path + sample_filename, 0)
sample_img = cv2.resize(sample_img, None, fx = expand_sample, fy = expand_sample)
kp_samp, des_samp = akaze.detectAndCompute(sample_img, None)

# 特徴量マッチング実行
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_temp, des_samp, k=2)

# マッチング精度が高いもののみ抽出
ratio = 0.6
good = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append(m)

if len(good)>5:
    src_pts = np.float32([ kp_temp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_samp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = template_img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    sample_img = cv2.polylines(sample_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good), 5))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(template_img, kp_temp,sample_img,kp_samp,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()
cv2.imwrite(result_path + result_name, result_img)
cv2.waitKey(0) 