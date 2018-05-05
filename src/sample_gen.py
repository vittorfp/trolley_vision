#!/usr/bin/env python
import cv2
import time
import numpy as np
from common import clock, mosaic

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


if __name__ == '__main__':
	
	cap = cv2.VideoCapture('../img/lenta.mp4'	)
	winH,winW = (75,244)
	i  = 0
	while( cap.isOpened() ):
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		print(i)
		for (x, y, window) in sliding_window(gray, stepSize=10, windowSize=(winW, winH)):

			if window.shape[0] != winH or window.shape[1] != winW:
				continue

	 		window = cv2.transpose(window)
			window = cv2.flip(window,flipCode=2)
			
			cv2.imwrite('../img/dataset2/img' + str(i) + '.jpg' , window )
			i += 1