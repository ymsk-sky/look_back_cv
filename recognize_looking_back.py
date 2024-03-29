# -*- coding: utf-8 -*-

import json
import os

import cv2
import numpy as np


GHOST_POS_W = 0
GHOST_POS_H = 0
WINDOW_NAME = "cap"


def main():
    config = load_config()

    cap = cv2.VideoCapture(config["camera"])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cv2.namedWindow(WINDOW_NAME)

    # 合成する画像を準備
    img_file = os.path.join(config["image_path"], config["image_file"])
    ghost_img = cv2.imread(img_file)
    gray_img = cv2.cvtColor(ghost_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # 顔検出準備
    cascade_file = os.path.join(config["cascade_path"], config["cascade_file"])
    cascade = cv2.CascadeClassifier(cascade_file)
    # 顔判定の元となるサイズ
    base_w, base_h = cap_w//4, cap_h//4
    # キャリブレーション用フラグ
    calibrating = False

    print(f"camera size: {cap_w}x{cap_h}")
    print(f"camera fps: {fps}")

    print("press q to quit")
    print("press c to calibrate face size")
    while 1:
        ret, frame = cap.read()
        if not ret:
            continue

        # 顔認識
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1)
        )
        if len(face_rects) > 0:
            # 最大の矩形を顔候補とする
            rect = face_rects[np.argmax([r[2] for r in face_rects])]
            # 基準の大きさより大きいなら顔と判定
            if rect[2] > base_w*0.8 or rect[3] > base_h*0.8:
                if calibrating:
                    rect_color = (0, 0, 255)
                else:
                    rect_color = (255, 0, 0)
                cv2.rectangle(
                    frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]),
                    rect_color, thickness=4
                )
            else:
                # 顔候補はあったが顔と判定されなかった場合: 合成
                appear_ghost(frame, cap_w, cap_h, ghost_img, mask, mask_inv)
        else:
            # 顔候補がなかった場合: 合成
            appear_ghost(frame, cap_w, cap_h, ghost_img, mask, mask_inv)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(fps)
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("c"):
            if calibrating:
                base_w = rect[2]
                base_h = rect[3]
            calibrating = not calibrating

    cap.release()
    cv2.destroyWindow(WINDOW_NAME)


def appear_ghost(frame, w, h, img, mask, mask_inv):
    # 画面上の暗い場所を探索
    # 画像をいい感じに重畳表示
    row, col = img.shape[:2]
    roi = frame[GHOST_POS_W:GHOST_POS_W+row,
                GHOST_POS_H:GHOST_POS_H+col]

    frame_background = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_foreground = cv2.bitwise_and(img, img, mask=mask)
    res = cv2.add(frame_background, img_foreground)
    frame[GHOST_POS_W:GHOST_POS_W+row, GHOST_POS_H:GHOST_POS_H+col] = res


def load_config(path="./conf"):
    file = os.path.join(path, "config.json")
    with open(file, "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    main()
