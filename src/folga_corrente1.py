import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


cap = cv2.VideoCapture('../img/video2.mp4')
i = 0
y = []
x = []
tam = []
while(cap.isOpened()):
	i += 1
	ret, frame = cap.read()
	if frame is None:
		break
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[40:250,70:450]
	frame = cv2.equalizeHist(frame)
	frame = adjust_gamma(frame, gamma=0.6)
	frame = cv2.equalizeHist(frame)
	frame = adjust_gamma(frame, gamma=0.8)
	frame = cv2.equalizeHist(frame)
	out   = frame.copy()
	frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,4)
	frame = cv2.Canny(frame, 900, 1000)
	frame = frame[80:120,50:300]
	# change into uint8
	#frame = cv2.convertScaleAbs(g)

	#_, frame = cv2.threshold(frame, 90,255	,cv2.THRESH_BINARY)
	y.append( np.sum( np.sum(frame) ) )
	x.append( i )
	#plt.plot(x,y)
	#plt.show()
	print(frame.shape)
	a = range(250)

	if y[i-1] < 250000:
		print('Buraco', i)
		linha = frame[20,:]

		dists = np.squeeze(np.argwhere(linha == 255))
		dif = np.diff(dists)
		mx = np.max(dif)

		idx_dif = np.argwhere(dif == mx)
		init = np.squeeze(dists[idx_dif])
		end = np.squeeze(dists[idx_dif+1])
		print(mx)
		if mx > 100:
			tam.append(mx)
			cv2.line(out,(init+50, 20+80),(end+50,20+80),(100,100,100), 3)
			
	else:
		if len(tam) != 0:
			plt.plot(tam)	
			plt.show()
		tam = []

	cv2.imshow("Image", frame)
	cv2.imshow("Image1", out)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	time.sleep(0.03)
