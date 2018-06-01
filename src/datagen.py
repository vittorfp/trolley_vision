import numpy as np
import pandas as pd
import imutils
import os
import time
import cv2

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

template_pos = cv2.cvtColor(cv2.imread('../img/template/template_arrastador_pos.jpg'), cv2.COLOR_BGR2GRAY)
template_pre = cv2.cvtColor(cv2.imread('../img/template/template_arrastador_pre.jpg'), cv2.COLOR_BGR2GRAY)

df = pd.DataFrame()
# Itera sobre os positivos
for root, dirs, files in os.walk("../img/arrastador_dataset/yes"):
	for filename in files:
		#print(filename)
		image = cv2.imread('../img/arrastador_dataset/yes/pre/'+filename)
		di = 'pre'
		if image is None:
			image = cv2.imread('../img/arrastador_dataset/yes/pos/'+filename)
			di = 'pos'

		#cv2.imshow("fdsf",image)
		#cv2.imshow("dsad",template)
		#cv2.waitKey(0)

		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		a = template_match(image,template_pre)
		b = template_match(image,template_pos)
		if(a is None) | (b is None):
			pass
		else:
			(maxVal1, maxLoc1) = a
			(maxVal2, maxLoc2) = b
			print({'label': 1, 'maxVal_pre': maxVal1, 'maxVal_pos': maxVal2 })
		line = pd.DataFrame({'label': 1, 'maxVal_pre': maxVal1, 'maxVal_pos': maxVal2, 'direction':di }, index = [0])
		df = df.append(line,ignore_index=True)


# Itera sobre os negativos
# Itera sobre os positivos
for root, dirs, files in os.walk("../img/arrastador_dataset/no"):
	for filename in files:
		print(filename)
		image = cv2.imread('../img/arrastador_dataset/no/'+filename)
		#cv2.imshow("fdsf",image)
		#cv2.imshow("dsad",template)
		#cv2.waitKey(0)

		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		a = template_match(image,template_pre)
		b = template_match(image,template_pos)
		if(a is None) | (b is None):
			pass
		else:
			(maxVal1, maxLoc1) = a
			(maxVal2, maxLoc2) = b
			print({'label': 1, 'maxVal_pre': maxVal1, 'maxVal_pos': maxVal2 })
		line = pd.DataFrame({'label': 0, 'maxVal_pre': maxVal1, 'maxVal_pos': maxVal2 , 'direction':di }, index = [0])
		df = df.append(line,ignore_index=True)

df.to_csv('data_arrast.csv')

