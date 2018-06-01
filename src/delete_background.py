import numpy as np
import cv2

cap = cv2.VideoCapture('../img/video2.mp4')
history = 10 
varThreshold = 20
bShadowDetection = False

fgbg = cv2.createBackgroundSubtractorMOG2(history, varThreshold, bShadowDetection)

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
 
    cv2.imshow('fgmask',frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',fgmask)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()