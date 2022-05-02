import numpy as np
import cv2
from sqlalchemy import within_group

CAMERA = 0
CASCADE_FILE = "./haarcascade_frontalface_default.xml"

def main():
    # カメラ読み込み
    capture = cv2.VideoCapture(CAMERA)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    window = "cap"
    cv2.namedWindow(window)

    cascade = cv2.CascadeClassifier(CASCADE_FILE)

    while(1):
        ret, frame = capture.read()
        if(not ret):
            break
        cv2.imshow(window, frame)

        key = cv2.waitKey(fps)
        if(key & 0xFF == ord('q')):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
