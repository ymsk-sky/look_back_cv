import numpy as np
import cv2

CAMERA = 0
CASCADE_FILE = "./haarcascade_frontalface_default.xml"
IMAGE_FILE = "./obake_resized.png"

def main():
    # カメラ読み込み
    capture = cv2.VideoCapture(CAMERA)
    # カメラオプション取得
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    cap_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
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

    # 顔判定の元となるサイズ
    base_w, base_h = 0, 0

    calibration_f = False
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

            # 顔サイズと同程度の場合に顔と判定
            if(rect[2] > base_w * 0.8 or rect[3] > base_h * 0.8):
                if(calibration_f):
                    # キャリブレーション中は判定矩形を表示
                    cv2.rectangle(frame, tuple(rect[0:2]),
                                         tuple(rect[0:2] + rect[2:4]),
                                         (255, 0, 0), thickness=4)
            # おばけ出現
            appear_obake(frame, img, rows, cols, mask, mask_inv, cap_w, cap_h)

        # 表示
        cv2.imshow(window, frame)

        key = cv2.waitKey(fps)
        if(key & 0xFF == ord('q')):
            break
        elif(key & 0xFF == ord('c')):
            if(calibration_f):
                base_w = rect[2]
                base_h = rect[3]
            # 再度押すまでキャリブレーションし続ける
            calibration_f = not calibration_f
        
        cnt += 1
        
    capture.release()
    cv2.destroyAllWindows()


def appear_obake(frame, img, rows, cols, mask, mask_inv, w, h):
    # 合成位置を決定
    x, y = int(w / 4), int(h / 4)
    roi = frame[x:x+rows, y:y+cols]

    frame_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(img, img, mask=mask)

    dst = cv2.add(frame_bg, img_fg)
    frame[x:x+rows, y:y+cols] = dst


if __name__ == "__main__":
    main()
