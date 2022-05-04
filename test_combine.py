import numpy as np
import cv2

IMAGE_FILE ="./obake_resized.png"
BACK_FILE ="./background.jpg"

img = cv2.imread(IMAGE_FILE)
background = cv2.imread(BACK_FILE)

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