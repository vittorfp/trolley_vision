import numpy as np
import argparse
import imutils
import glob
import cv2
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=False, help="Path to template image")
ap.add_argument("-i", "--images", required=False, help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize", help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

# load the image image, convert it to grayscale, and detect edges
template_malha = cv2.imread('../img/template/template_malha.jpg')
template_elo = cv2.imread('../img/template/template_elo.jpg')
template_arrastador = cv2.imread('../img/template/template_malha_arrastador.jpg')
template_trolley = cv2.imread('../img/template/template1.jpg')

template_malha = cv2.cvtColor(template_malha, cv2.COLOR_BGR2GRAY)
template_elo = cv2.cvtColor(template_elo, cv2.COLOR_BGR2GRAY)
template_arrastador = cv2.cvtColor(template_arrastador, cv2.COLOR_BGR2GRAY)
template_trolley = cv2.cvtColor(template_trolley, cv2.COLOR_BGR2GRAY)

t = 0.01

def searchTemplate(image,template):
	(tH, tW) = template.shape[:2]
	found = None

	for scale in np.linspace(0.2, 1.0, 30)[::-1]:

		resized = imutils.resize(image, width = int(image.shape[1] * scale))
		r = image.shape[1] / float(resized.shape[1])
		
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

		resized = cv2.GaussianBlur(resized,(3,3),0)
		edged = cv2.Canny(resized, 30, 50)
	
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
		
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r,result) 	
	(maxVal, maxLoc, r,result) = found
	return(maxVal, maxLoc, r, result) 

font = cv2.FONT_HERSHEY_SIMPLEX
    	
fh = open("dataset_manager/correntes.txt", "r+") 

for i in range(1300,3252):

	image_corrente = cv2.imread('../img/dataset3/img' + str(i) + '.jpg' )
	image_trolley = cv2.imread('../img/dataset2/img' + str(i) + '.jpg' )
	
	# Procura os tipos das correntes
	(  maxVal_elo  , maxLoc, r, res_elo) = searchTemplate( image_corrente, template_elo )
	( maxVal_malha , maxLoc, r, res_malha) = searchTemplate( image_corrente, template_malha )
	( maxVal_arrastador , maxLoc, r, res_arrastador) = searchTemplate( image_corrente, template_arrastador )
	
	# Procura o trolley
	( maxVal_trolley , maxLoc, r, res_trolley) = searchTemplate( image_trolley, template_trolley )

	if(maxVal_malha > 1.2e7):
	#if(maxVal_malha/7210549 > maxVal_elo/4291963) and (maxVal_malha/7210549 > maxVal_arrastador/2811146):
	#if(maxVal_malha > maxVal_elo) and (maxVal_malha > maxVal_arrastador):
		print('Malha')
		cv2.putText(image_corrente,'Malha',(40,40), font, 2,(255,0,0),2,cv2.LINE_AA)

	else:
		if(maxVal_elo > 0.7e7):
		#if(maxVal_elo/4291963 > maxVal_malha/7210549) and (maxVal_elo/4291963 > maxVal_arrastador/2811146):
		#if(maxVal_elo > maxVal_malha) and (maxVal_elo > maxVal_arrastador):
			print('Elo')
			cv2.putText(image_corrente,'Elo',(40,40), font, 2,(255,0,0),2,cv2.LINE_AA)
	#if(maxVal_arrastador/2811146 > maxVal_elo/4291963) and (maxVal_arrastador > maxVal_malha/7210549):
	#	print('Arrastador')

	#(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	#(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
	#print(maxVal)

	#fh.write( str(maxVal_elo) + ',' + str(maxVal_malha) + ',' + str(maxVal_arrastador) + ',' + str(maxVal_trolley) +'\n' )
	
	
	cv2.imshow("Template", res_elo)
	cv2.imshow("correntes", image_corrente)
	cv2.imshow("trolley", image_trolley)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	#time.sleep(t)

try:
	pass
except:
	print('Exeption')
	fh.close()
fh.close()