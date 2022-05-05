import numpy as np
import cv2

IMAGE_FILE ="./obake_resized.png"
BACK_FILE ="./background.jpg"
G_FILE ="./gradation.jpg"

img = cv2.imread(IMAGE_FILE, -1)  # アルファ値を含んで読み込む
background = cv2.imread(BACK_FILE)
g = cv2.imread(G_FILE)
g2 = cv2.resize(g, img.shape[1::-1])  # リサイズ
g3 = cv2.cvtColor(g2, cv2.COLOR_BGR2GRAY)  # グレースケール
g3 = cv2.bitwise_not(g3)  # 反転

rows, cols, _ = img.shape

img2 = img.copy()
for i in range(rows):
    for j in range(cols):
        if(img[i][j][3] == 0):
            pass
        else:
            img2[i][j][3] = g3[i][j]

cv2.imwrite("img2.png", img2)

"""
[
    [[R,G,B,A],...,[R,G,B,A]],
    [[R,G,B,A],...,[R,G,B,A]],
    ...
    [[R,G,B,A],...,[R,G,B,A]],
    [[R,G,B,A],...,[R,G,B,A]]
]
"""

"""
rows, cols, _ = img.shape
w, h, _ = background.shape

# Make a mask
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

roi = background[0:rows, 0:cols]
background_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img_fg = cv2.bitwise_and(img, img, mask=mask)

dst = cv2.add(background_bg, img_fg)
background[0:rows, 0:cols] = dst

cv2.imwrite("test.jpg", background)
"""
