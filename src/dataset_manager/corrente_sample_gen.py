#!/usr/bin/env python
import cv2
import time
import numpy as np

if __name__ == '__main__':
	
	cap = cv2.VideoCapture('../../img/lenta.mp4')
	i  = 0

	# Coloca para abrir a partir do meio do video, que esta' com o angulo
	# de filmagem melhor que o do inicio

	cap.set(1,1308);
	while( cap.isOpened() ):
		ret, frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[170:350,200:430]
		print(i)
		window = cv2.GaussianBlur(frame,(5,5),0)
		window = cv2.Canny(window, 30,50)
		#cv2.imshow("Imagem", cv2.Canny(window, 30,50))
		#key = cv2.waitKey(0)

		#while not(key == 113 or key == 116 or key == 102):
		#	print("Opcao invalida")
		#	key = cv2.waitKey(0)
		
		# q	
		#if(key == 113):
		#	break

		cv2.imwrite('../../img/dataset3/img' + str(i) + '.jpg' , window)
		i += 1