import numpy as np
import cv2
import tkinter
from matplotlib import pyplot as plt
from PIL import Image, ImageTk


# 画像の外周を0で埋めるメソッドです
def padding_zeros(image,sz):
    import numpy as np
    m,n = np.shape(image)
    padded_image = np.full((m+sz*2,n+sz*2),255)
    padded_image[sz:-sz,sz:-sz] = image
    return np.uint8(padded_image)

# 外周1行1列を除くメソッドです。
def unpadding(image,sz):
    return image[sz:-sz, sz:-sz]



im = cv2.imread('board.jpg')

imb = np.copy(im)

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)#[[max(b[1],b[2]) for b in a] for a in cv2.cvtColor(im,cv2.COLOR_BGR2HSV_FULL)]
#print(pboard)

plt.figure()
plt.imshow(imgray, cmap="gray")



adpsize = int(im.shape[0]/30) + 1 - int(im.shape[0]/30) %2
thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,adpsize,50)
#thresh = cv2.bitwise_not(thresh)
plt.figure()
plt.imshow(thresh, cmap="gray")
thresh = padding_zeros(thresh,10)

neiborhood = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],
             np.uint8)
for i in range(100):
    thresh = cv2.erode(thresh,neiborhood,iterations=10)
    thresh = cv2.dilate(thresh,neiborhood,iterations=10)

thresh = unpadding(thresh,10)

plt.figure()
plt.imshow(thresh, cmap="gray")

#plt.figure()
#plt.imshow(thinning(thresh), cmap="gray")

thresh = cv2.bitwise_not(thresh)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

apxcnt = []

for cnt in contours:
    apxcnt.append(cv2.convexHull(cnt))

#scont = [ (menseki(cv2.minAreaRect(cnt)) ,cnt) for cnt in contours for menseki in [lambda x:x.width*x.height]]
#scont.sort(key=lambda x:-x[0])
#contours = np.array([cnt[1] for cnt in scont])
print(len(contours))


img = cv2.drawContours(im, apxcnt, -1, (0,255,0), 2)

plt.figure()
plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
plt.show()


#root = tkinter.Tk()
# im = np.copy(imb)
# image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )# RGBからPILフォーマットへ変換
# image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換
# canvas = tkinter.Canvas(root, width=img.shape[1], height=img.shape[0]) # Canvas作成
# canvas.pack()
# imgcnv = canvas.create_image(0, 0, image=image_tk, anchor='nw') # ImageTk 画像配置
# print(len(contours))
# def drawImg():
#     global idx
#     global canvas
#     global imgcnv
#     global im
#     global contours
#     idx += 1
#     print(contours[idx])
#     img = cv2.drawContours(im, contours, idx, (0,255,0), 2)
#     im = np.copy(imb)
#     image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )# RGBからPILフォーマットへ変換
#     image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換
#     canvas.itemconfig(imgcnv, image=image_tk) # ImageTk 画像配置
#     canvas.photo = image_tk

# nxButton = tkinter.Button(root, text=">", command=drawImg)
# nxButton.place(x=0,y=0)

# root.mainloop()