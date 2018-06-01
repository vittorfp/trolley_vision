import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def template_match(img,template):
	found = None
	#print(template.shape)
	#print(img.shape)
	(tH, tW) = template.shape[:2]
			
	
	img = cv2.GaussianBlur(img,(3,3),0)
	#edged = cv2.Canny(resized, 30, 50)

	result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
	(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
	
	if found is None or maxVal > found[0]:
		found = (maxVal, maxLoc)
	
	return found

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
j = 0

history = 5
varThreshold = 10
bShadowDetection = False

init_x = 170
init_y = 120
end_x = 300
end_y = 220

fgbg = cv2.createBackgroundSubtractorMOG2(history, varThreshold, bShadowDetection)
t = 0

template_pre = cv2.cvtColor(cv2.imread('../img/template/template_arrastador_pre.jpg'), cv2.COLOR_BGR2GRAY)
template_pos = cv2.cvtColor(cv2.imread('../img/template/template_arrastador_pos.jpg'), cv2.COLOR_BGR2GRAY)
font = cv2.FONT_HERSHEY_SIMPLEX

cap.set(cv2.CAP_PROP_POS_FRAMES, 700)

while(cap.isOpened()):
	ret, frame = cap.read()
	if frame is None:
		break
	
	out = frame.copy()[init_x:end_x,init_y:end_y]
	frame = frame[init_x:end_x,init_y:end_y,:]
	fm =frame.copy()
	out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	out = adjust_gamma(out, gamma=0.8)
	out = cv2.GaussianBlur(out,(7,7),0)
	_, out = cv2.threshold(out,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#_, out = cv2.threshold(out, 100,255	,cv2.THRESH_BINARY)

	pt = out.copy()
	e = fgbg.apply(out)
	
	#cv2.imshow('fgmask',e)
	#cv2.imshow("gaussiano", out)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	#out = cv2.equalizeHist(out)
	#out = cv2.adaptiveThreshold(out,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,5)
	#out = cv2.bitwise_not(out)
	
	a = template_match(pt,template_pre)
	b = template_match(pt,template_pos)
	if(a is None) | (b is None):
		pass
	else:
		(maxVal_pre, maxLoc1) = a
		(maxVal_pos, maxLoc2) = b
	angles = []

	if(  maxVal_pos > 1.4e8 ) | ( maxVal_pre > 1.05e8 ):
		out = cv2.Canny(out,120,150,apertureSize = 3)
		lines = cv2.HoughLinesP(image=out, rho=1, theta = np.pi/180, threshold=20,lines=np.array([]), minLineLength=25,maxLineGap=10)
		
		
		vectors = [[0,0]]
		if lines is None:
			pass
		else:
			a,b,c = lines.shape
			for i in range(a):
				cv2.line(frame, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 2, cv2.LINE_AA)
				vec = [[abs(lines[i][0][2] - lines[i][0][0]), abs(lines[i][0][3]-lines[i][0][1]) ]]
				vectors = np.append(vectors,vec,axis= 0)
				
			
			vectors = np.squeeze(vectors[1:])
			#print(vectors)
			#print('\n\n')
			
			angles = []
			for vec1 in vectors:
				for vec2 in vectors:
					angle = np.arccos( np.inner(vec1,vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2)))
					angles.append(angle*57.2958)
			#angles = np.append(angles[angles <= 90] , np.absolute( np.array(angles[angles > 90]) - 90-90 ))
			mx = max(angles)
			cv2.putText(frame,str(mx),(40,110), font, 1,(255,0,0),2,cv2.LINE_AA)
			print('Arrastador',mx)
			t = 1

	cv2.imshow("Image", frame)
	#cv2.imshow("Image1", out)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if len(angles) > 0:
		if  (max(angles) > 85) & (max(angles) < 95):
			pass
			j += 1
			print(max(angles))
			#cv2.imwrite('../img/arrastador_dataset/' + str(j) + '.jpg' , out)
			#time.sleep(0.0)
		else:
			time.sleep(0.00)
	time.sleep(t)
	t = 0.000
