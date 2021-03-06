import cv2
import time
import numpy as np
import pandas as pd
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
tam1 = []
tam2 = []
tam3 = []

pix1 = []
pix2 = []
pix3 = []

detection = 0

df = pd.DataFrame()
while(cap.isOpened()):
	i += 1
	ret, frame = cap.read()
	if frame is None:
		break
	out   = frame.copy()[40:250,70:450]
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[40:250,70:450]
	frame = cv2.equalizeHist(frame)
	frame = adjust_gamma(frame, gamma=0.6)
	frame = cv2.equalizeHist(frame)
	frame = adjust_gamma(frame, gamma=0.8)
	frame = cv2.equalizeHist(frame)
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
	#print(frame.shape)
	a = range(250)

	if y[i-1] < 250000:
		print('Buraco', i)
		linha1 = frame[15,:]
		linha2 = frame[20,:]
		linha3 = frame[25,:]
		
		dists = np.squeeze(np.argwhere(linha1 == 255))
		dif = np.diff(dists)
		mx = np.max(dif)
		idx_dif = np.argwhere(dif == mx)
		init = np.squeeze(dists[idx_dif])
		end = np.squeeze(dists[idx_dif+1])
		print(mx)
		
		if mx > 100:
			tam1.append(mx)
			pix1.append(init)
			line = pd.DataFrame({'tamanho': mx, 'inicio': init, 'altura': 'low','detection':detection}, index = [0])
			df = df.append(line,ignore_index=True)
			cv2.line(out,(init+50, 15+80),(end+50,15+80),(255,0,0), 2)

		dists = np.squeeze(np.argwhere(linha2 == 255))
		dif = np.diff(dists)
		mx = np.max(dif)
		idx_dif = np.argwhere(dif == mx)
		init = np.squeeze(dists[idx_dif])
		end = np.squeeze(dists[idx_dif+1])
		print(mx)

		if mx > 100:
			tam2.append(mx)
			pix2.append(init)
			line = pd.DataFrame({'tamanho': mx, 'inicio': init, 'altura': 'middle','detection':detection}, index = [0])
			df = df.append(line,ignore_index=True)
			cv2.line(out,(init+50, 20+80),(end+50,20+80),(0,0,255), 2)

		dists = np.squeeze(np.argwhere(linha3 == 255))
		dif = np.diff(dists)
		mx = np.max(dif)
		idx_dif = np.argwhere(dif == mx)
		init = np.squeeze(dists[idx_dif])
		end = np.squeeze(dists[idx_dif+1])
		print(mx)

		if mx > 100:
			tam3.append(mx)
			pix3.append(init)
			line = pd.DataFrame({'tamanho': mx, 'inicio': init, 'altura': 'high','detection':detection}, index = [0])
			df = df.append(line,ignore_index=True)
			
			cv2.line(out,(init+50, 25+80),(end+50,25+80),(0,255,0), 2)
			
		#print(df)
	else:
		detection += 1
		if len(tam1) != 0:
			pass
			plt.plot(pix1,tam1)	
			plt.plot(pix2,tam2)	
			plt.plot(pix3,tam3)	
			plt.show()

		tam1 = []
		tam2 = []
		tam3 = []
		pix1 = []
		pix2 = []
		pix3 = []

	cv2.imshow("Image", frame)
	cv2.imshow("Image1", out)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	time.sleep(0.03)

df.to_csv('tamanhos.csv')