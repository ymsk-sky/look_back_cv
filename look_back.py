import numpy as np
import cv2
from sqlalchemy import within_group

CAMERA = 0
CASCADE_FILE = "./haarcascade_frontalface_default.xml"
IMAGE_FILE = "./obake_resized.png"

def main():
    # カメラ読み込み
    capture = cv2.VideoCapture(CAMERA)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    window = "cap"
    cv2.namedWindow(window)

    # 顔検出カスケードファイル読み込み
    cascade = cv2.CascadeClassifier(CASCADE_FILE)

    # 合成画像を取得
    img = cv2.imread(IMAGE_FILE)
    rows, cols, _ = img.shape
    # グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # マスク作成
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # フレーム数管理
    cnt = 0

    while(1):
        ret, frame = capture.read()
        if(not ret):
            break
        
        # 10フレームで1回の顔認識処理
        if(cnt % 10 == 0):
            cnt = 0
            # グレースケール
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 顔をすべて認識
            facerects = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                 minNeighbors=1, minSize=(1, 1))
        
        # ひとつでも顔が検出されたとき
        if(len(facerects) > 0):
            # 矩形が最大のものを顔候補とする
            rect = facerects[np.argmax([r[2] for r in facerects])]

            # おばけ出現
            appear_obake(frame, img, rows, cols, mask, mask_inv)

        # 表示
        cv2.imshow(window, frame)

        key = cv2.waitKey(fps)
        if(key & 0xFF == ord('q')):
            break
        
        cnt += 1
        
    capture.release()
    cv2.destroyAllWindows()


def appear_obake(frame, img, rows, cols, mask, mask_inv):
    # 合成位置を決定
    roi = frame[0:rows, 0:cols]

    frame_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(img, img, mask=mask)

    dst = cv2.add(frame_bg, img_fg)
    frame[0:rows, 0:cols] = dst


if __name__ == "__main__":
    main()
