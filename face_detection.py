# just recognises face or eyes depending on application, and plots a rectangle around object on webcam feed

import cv2
import dlib

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
video_capture = cv2.VideoCapture(0)        # 0 because it is capturing feed from main webcam linked to device

while True:
    ret,frame = video_capture.read()
    image_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = face_detector.detectMultiScale(image_grey, minSize=(105,106))
    eye_detections = face_detector.detectMultiScale(image_grey)

    for (x,y, w,h) in eye_detections:
        print('\n--------------', eye_detections)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255), 2)

    cv2.imshow('Video',frame)
    if cv2.waitKey(delay=2) & 0xFF=='Q':
        break