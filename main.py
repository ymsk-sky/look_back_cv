# -*- coding: utf-8 -*-

import numpy as np
import cv2

def main():
    # カメラ読み込み
    cap = cv2.VideoCapture(2)
    # 合成画像読み込み
    img = cv2.imread('obake_resized.png')

    # 顔検出カスケードファイル読み込み
    cascade_path = "/usr/local/var/pyenv/versions/3.7.1/lib/python3.7/site-packages/cv2/data/"
    cascade_file = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path+cascade_file)

    # 矩形用の色
    color = (0, 0, 255)

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
            facerect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

        # ひとつでも検出されている場合
        if(len(facerect) > 0):
            # 矩形が最も大きいものを顔とする
            rect = facerect[np.argmax([r[2] for r in facerect])]
            # 矩形を描く
            cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=4)
        else:
            pass

        # 表示
        cv2.imshow('cap', frame)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
        cnt += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
