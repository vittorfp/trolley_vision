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
 
fh = open("dataset_manager/labels.txt", "r") 
num = len(fh.readlines())
fh.close()
trolleys = 0

# load the image image, convert it to grayscale, and detect edges
template = cv2.imread('../img/template/template1.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
(tH, tW) = template.shape[:2]

cv2.imshow("Template", template)
trolley_anterior = 0
circulo_anterior = 0
# loop over the images to find the template in
historic = [0 ,0 ,0 ,0 ,0 ,0,0 ,0 ,0 ,0 ,0 ,0]
historic_trolley = [0 ,0 ,0 ,0 ,0 ,0,0 ,0 ,0 ,0 ,0 ,0]
def new_frame(vec,num):
	vec.pop(0)
	vec.append(num)
	return vec

for i in range(1308,num+1307):

	image = cv2.imread('../img/dataset2/img' + str(i) + '.jpg' )
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found = None
 
	# loop over the scales of the image
	for scale in np.linspace(0.2, 1.0, 30)[::-1]:

		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])
 
		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break
		resized = cv2.GaussianBlur(resized,(3,3),0)
		edged = cv2.Canny(resized, 30, 50)
	
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
 		
		if args.get("visualize", False):
			clone = np.dstack([edged, edged, edged])
			cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
				(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
			cv2.imshow("Visualize", clone)
			cv2.waitKey(0)
 
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)
 
	(maxVal, maxLoc, r) = found
	#f.write( str(maxVal) + '\n' )
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

	# define o limiar de deteccao do trolley
	if( maxVal > 0.65e7 ):
		historic_trolley = new_frame(historic_trolley,1)

		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		local = gray[startY:endY, startX:endX]
		circles = cv2.HoughCircles(local, cv2.HOUGH_GRADIENT, 4,100, maxRadius = 45, minRadius = 40)
		
		if circles is not None:
			circles = np.round(circles[0, :]).astype("int")
			for (x, y, r) in circles:
				cv2.circle(image, (x + startX, y + startY), r, (0, 255, 0), 4)
				cv2.rectangle(image, (x + startX - 5, y + startY - 5), (x + startX + 5, y + startY + 5), (0, 128, 255), -1)

			circulo_anterior += 1
			historic = new_frame(historic,1)
		else:
			historic = new_frame(historic,0)

		trolley_anterior += 1
		t = 0.01
	
	else:
		
		historic = new_frame(historic,0)
		historic_trolley = new_frame(historic_trolley,0)
		
		if(trolley_anterior >= 1):
			if(circulo_anterior == 1):
				
				if(sum(historic) != 0):
					if(sum(historic_trolley) > 3):
						trolleys += 1
						print('OK!',trolleys)
			else:
				if(sum(historic) == 0):
					if(sum(historic_trolley) > 3):
						print( 'Potencial problema!' )

		trolley_anterior = 0
		circulo_anterior = 0
		t=0.01

	cv2.imshow("Image", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	time.sleep(t)
