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


cap = cv2.VideoCapture('../img/video4.mp4')
df = pd.DataFrame()
i = 0

init_x = 170
init_y = 120
end_x = 300
end_y = 220

while(cap.isOpened()):
	i += 1
	ret, frame = cap.read()
	if frame is None:
		break
	
	out = frame.copy()[init_x:end_x,init_y:end_y]
	frame = frame[init_x:end_x,init_y:end_y,:]
	out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	out = adjust_gamma(out, gamma=0.8)
	out = cv2.GaussianBlur(out,(7,7),0)
	_, out = cv2.threshold(out,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#_, out = cv2.threshold(out, 100,255	,cv2.THRESH_BINARY)
	
	cv2.imshow("gaussiano", out)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	#out = cv2.equalizeHist(out)
	#out = cv2.adaptiveThreshold(out,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,5)
	#out = cv2.bitwise_not(out)
	
	out = cv2.Canny(out,120,150,apertureSize = 3)
	lines = cv2.HoughLinesP(image=out, rho=1, theta=np.pi/180, threshold=20,lines=np.array([]), minLineLength=20,maxLineGap=30)
	
	vectors = [[0,0]]
	if lines is None:
		pass
	else:
		a,b,c = lines.shape
		for i in range(a):
			cv2.line(frame, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
			vec = [[lines[i][0][2] - lines[i][0][0], lines[i][0][3]-lines[i][0][1] ]]
			vectors = np.append(vectors,vec,axis= 0)
			
		
		vectors = np.squeeze(vectors[1:])
		#print(vectors)
		#print('\n\n')
		
		angles = []
		for vec1 in vectors:
			for vec2 in vectors:
				angle = np.arccos( np.inner(vec1,vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2)))
				angles.append(angle*57.2958)
		
		
		

	cv2	.imshow("Image", frame)
	cv2.imshow("Image1", out)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if  (max(angles) > 88) & (max(angles) < 92):
		print(max(angles))
		time.sleep(0.5)
	else:
		time.sleep(0.00)

