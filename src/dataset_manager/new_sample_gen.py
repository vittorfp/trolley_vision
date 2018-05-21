#!/usr/bin/env python
import cv2
import time
import numpy as np

cap = cv2.VideoCapture('../../img/lenta.mp4')
i  = 0
while( cap.isOpened() ):
	ret, frame = cap.read()
	window = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[0:300,200:340]
	print(i)
	cv2.imwrite('../../img/dataset2/img' + str(i) + '.jpg' , window )
	i += 1