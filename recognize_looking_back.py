# -*- coding: utf-8 -*-

import json
import os

import cv2
import numpy as np


def main():
    config = load_config()

    cap = cv2.VideoCapture(config["camera"])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    window = "cap"
    cv2.namedWindow(window)

    # 顔検出準備
    cascade_file = os.path.join(config["cascade_path"], config["cascade_file"])
    cascade = cv2.CascadeClassifier(cascade_file)

    while 1:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1)
        )
        if len(face_rects) > 0:
            rect = face_rects[np.argmax([r[2] for r in face_rects])]
            cv2.rectangle(
                frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]),
                (255, 0, 0), thickness=4
            )

        cv2.imshow(window, frame)

        key = cv2.waitKey(fps)
        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyWindow(window)


def load_config(path="./conf"):
    file = os.path.join(path, "config.json")
    with open(file, "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    main()
