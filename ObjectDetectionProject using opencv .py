
# Color Detection Project
import numpy as np
import cv2
import os

webcam = cv2.VideoCapture(0)

while True:
    ret,image = webcam.read()
    hsvcimage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lowerlimit = np.array([22, 100, 100])
    upperlimit = np.array([30, 255, 255])
    mask = cv2.inRange(hsvcimage,lowerlimit,upperlimit)


    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x1,y1,h,w = cv2.boundingRect(cnt)
        cv2.rectangle(image,(x1,y1),(x1+h,y1+w),(0,255,0),3)

    cv2.imshow('Window1', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



















