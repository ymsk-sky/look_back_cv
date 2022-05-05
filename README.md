# LOOK BACK CV
Webカメラを使用した場面（オンライン会議、オンライン飲み会など）において`幽霊`が現れるドッキリを行なう。
OpenCVによる顔検出を行ない、正面を向いているときに`幽霊`を出現させ、振り向いたときはその振り向きを検出し`幽霊`を消す挙動をする。

look back: 振り返る

# DEMO

TODO: demo動画(gif)

# Installation

## Libraries
- openCV
- numpy

```bash
$ pip install numpy
$ pip install opencv-python
$ pip install opencv-contrib-python
```

## Cascade file
<a href="https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml" target="_blank">haarcascade_frontalface_default.xml</a>

# Usage

## 実行

```bash
$ git clone https://github.com/ymsk-sky/look_back_cv.git
$ cd look_back_cv
$ python look_back.py
```
