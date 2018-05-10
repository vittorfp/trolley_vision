import numpy as np
import argparse
import imutils
import glob
import cv2
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=False, help="Path to template image")
ap.add_argument("-i", "--images", required=True, help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize", help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())
 
#fh = open("dataset_manager/labels.txt", "r") 
#num = len(fh.readlines())
#fh.close()

trolleys = 0

# load the image image, convert it to grayscale, and detect edges
template_malha = cv2.imread('../img/template/template_malha.jpg')
template_elo = cv2.imread('../img/template/template_elo.jpg')

template_malha = cv2.cvtColor(template_malha, cv2.COLOR_BGR2GRAY)
template_elo = cv2.cvtColor(template_elo, cv2.COLOR_BGR2GRAY)

(tH_m, tW_m) = template_malha.shape[:2]
(tH_e, tW_e) = template_elo.shape[:2]

cv2.imshow("Template", template)

def new_frame(vec,n):
	vec.pop(0)
	vec.append(n)
	return vec

for i in range(1308,3000):

	image = cv2.imread('../img/dataset3/img' + str(i) + '.jpg' )
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found = None
 
	for scale in np.linspace(0.2, 1.0, 30)[::-1]:

		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])
 		
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

		resized = cv2.GaussianBlur(resized,(3,3),0)
		edged = cv2.Canny(resized, 30, 50)
	
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
 		
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)
 	
 	(maxVal, maxLoc, r) = found

	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
	print(maxVal)

	t = 0.01
	# define o limiar de deteccao do trolley
	if( maxVal > 1e7 ):
		t = 0.1
		print("elo")
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)	
	
	cv2.imshow("Image", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	time.sleep(t)
