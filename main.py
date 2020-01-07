# -*- coding: utf-8 -*-
"""
透過画像の合成は公式チュートリアルを参考
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html#bitwise-operations
"""

import numpy as np
import cv2

# カメラ番号
CAMERA = 2

# 顔検出カスケードファイル読み込み
cascade_path = "/usr/local/var/pyenv/versions/3.7.1/lib/python3.7/site-packages/cv2/data/"
cascade_file = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path+cascade_file)


def main():
    # カメラ読み込み
    cap = cv2.VideoCapture(CAMERA)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 合成画像読み込み
    img = cv2.imread('obake_resized.png')

    # 合成する画像情報を取得
    rows, cols, channels = img.shape
    # グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # マスク作成
    ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # 顔判定の元となるサイズ
    base_width = 0
    base_height = 0

    calibration_flag = False
    cnt = 0
    while(1):
        ret, frame = cap.read()
        if(not ret):
            break

        # 10フレームに一度の処理とする
        if(cnt%10 == 0):
            # グレースケール
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # すべての顔（らしきところを）認識
            facerect = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                minNeighbors=1, minSize=(1, 1))

        # ひとつでも検出されている場合
        if(len(facerect) > 0):
            # 矩形が最も大きいものを顔候補とする
            rect = facerect[np.argmax([r[2] for r in facerect])]

            # キャリブレーションしたサイズと同程度の場合に顔と判定
            if(rect[2] > base_width*0.8 or rect[3] > base_height*0.8):
                if(calibration_flag):
                    cv2.rectangle(frame, tuple(rect[0:2]),
                                         tuple(rect[0:2]+rect[2:4]),
                                         (255, 0, 0), thickness=4)

                # 顔検出時（正面を向いているとき）におばけ出現 <-> 振り向くと消える
                appear_obake(frame, img, rows, cols, mask, mask_inv)

        # 表示
        cv2.imshow('cap', frame)

        if(cv2.waitKey(1) & 0xFF == ord('o')):
            if(calibration_flag):
                base_width = rect[2]
                base_height = rect[3]
            calibration_flag = not calibration_flag

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
        cnt += 1

    cap.release()
    cv2.destroyAllWindows()


def appear_obake(frame, img, rows, cols, mask, mask_inv):
    # 合成する位置を決める
    roi = frame[0:rows, 0:cols]

    frame_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(img, img, mask=mask)

    dst = cv2.add(frame_bg, img_fg)
    frame[0:rows, 0:cols] = dst


if __name__ == '__main__':
    main()
