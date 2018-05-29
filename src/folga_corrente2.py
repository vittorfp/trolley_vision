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


cap = cv2.VideoCapture('../img/video3.mp4')
i = 0
y = []
x = []
tam1 = []
tam2 = []
tam3 = []
tam4 = []
while(cap.isOpened()):
	i += 1
	ret, frame = cap.read()
	if frame is None:
		break
	out   = frame.copy()[100:300,50:470]
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[100:300,50:470]
	frame = cv2.equalizeHist(frame)
	frame = adjust_gamma(frame, gamma=0.6)
	frame = cv2.equalizeHist(frame)
	frame = adjust_gamma(frame, gamma=0.8)
	frame = cv2.equalizeHist(frame)
	frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,4)
	frame = cv2.Canny(frame, 900, 1000)
	frame = frame[70:100,50:300]
	# change into uint8
	#frame = cv2.convertScaleAbs(g)

	#_, frame = cv2.threshold(frame, 90,255	,cv2.THRESH_BINARY)
	y.append( np.sum( np.sum(frame) ) )
	x.append( i )
	#plt.plot(x,y)
	#plt.show()
	#print(frame.shape)
	print(y[i-1])
	if y[i-1] < 235000:
		print('Buraco', i)
		linha1 = frame[10,:]
		linha2 = frame[15,:]
		linha3 = frame[20,:]
		linha4 = frame[25,:]
		
		dists = np.squeeze(np.argwhere(linha1 == 255))
		dif = np.diff(dists)
		mx = np.max(dif)
		idx_dif = np.argwhere(dif == mx)
		init = np.squeeze(dists[idx_dif])
		end = np.squeeze(dists[idx_dif+1])
		#print(mx)
		
		if mx > 100:
			tam1.append(mx)
			cv2.line(out,(init+50, 10+70),(end+50,10+70),(255,0,0), 2)
		else:
			tam1.append(0)
		dists = np.squeeze(np.argwhere(linha2 == 255))
		dif = np.diff(dists)
		mx = np.max(dif)
		idx_dif = np.argwhere(dif == mx)
		init = np.squeeze(dists[idx_dif])
		end = np.squeeze(dists[idx_dif+1])
		#print(mx)

		if mx > 100:
			tam2.append(mx)
			cv2.line(out,(init+50, 15+70),(end+50,15+70),(0,0,255), 2)
		else:
			tam2.append(0)			
		dists = np.squeeze(np.argwhere(linha3 == 255))
		dif = np.diff(dists)
		mx = np.max(dif)
		idx_dif = np.argwhere(dif == mx)
		init = np.squeeze(dists[idx_dif])
		end = np.squeeze(dists[idx_dif+1])
		#print(mx)

		if mx > 100:
			tam3.append(mx)
			cv2.line(out,(init+50, 20+70),(end+50,20+70),(0,255,0), 2)
		else:
			tam3.append(0)

		dists = np.squeeze(np.argwhere(linha4 == 255))
		dif = np.diff(dists)
		mx = np.max(dif)
		idx_dif = np.argwhere(dif == mx)
		init = np.squeeze(dists[idx_dif])
		end = np.squeeze(dists[idx_dif+1])
		#print(mx)

		if mx > 100:
			tam4.append(mx)
			cv2.line(out,(init+50, 25+70),(end+50,25+70),(0,255,255), 2)
		else:
			tam4.append(0)
	else:	
		if len(tam1) != 0:
			plt.plot(tam1)	
			plt.plot(tam2)	
			plt.plot(tam3)	
			plt.plot(tam4)	
			plt.show()
		tam1 = []
		tam2 = []
		tam3 = []
		tam4 = []

	cv2.imshow("Image", frame)
	cv2.imshow("Image1", out)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	time.sleep(0.03)
