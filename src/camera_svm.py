#!/usr/bin/env python

import cv2
import time
import numpy as np

SZ = 20
wdt = 75
hei = 244

CLASS_N = 2

# local modules
from common import clock, mosaic

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def load_samples():
	samples_img = []
	for i in range(9):
		samples_img.append(cv2.imread('../img/dataset/img'+ str(i + 1) + '.jpg', 0))
	
	labels = np.array([1,1,1,1,1,0,0,0,0])
	return samples_img, labels

def svmInit(C=12.5, gamma=0.50625):
	# Set up SVM for OpenCV 3
	svm = cv2.ml.SVM_create()
	# Set SVM type
	svm.setType(cv2.ml.SVM_C_SVC)
	# Set SVM Kernel to Radial Basis Function (RBF) 
	svm.setKernel(cv2.ml.SVM_RBF)
	# Set parameter C
	svm.setC(C)
	# Set parameter Gamma
	svm.setGamma(gamma)  
	return svm

def svmTrain(model, samples, responses):
	model.train(samples, cv2.ml.ROW_SAMPLE, responses)
	return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()


def svmEvaluate(model, digits, samples, labels):
	predictions = svmPredict(model, samples)
	accuracy = (labels == predictions).mean()
	print('Percentage Accuracy: %.2f %%' % (accuracy*100))

	confusion = np.zeros((CLASS_N, CLASS_N), np.int32)
	for i, j in zip(labels, predictions):
		confusion[int(i), int(j)] += 1
	print('confusion matrix:')
	print(confusion)

	vis = []
	for img, flag in zip(digits, predictions == labels):
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		if not flag:
			img[...,:2] = 0
		
		vis.append(img)
	return mosaic(25, vis)


def preprocess_simple(digits):
	return np.float32(digits).reshape(-1, SZ*SZ) / 255.0


def get_hog() : 
	winSize = (75,244)
	blockSize = (15,20)
	blockStride = (6,8)
	cellSize = (5,5)
	nbins = 9
	derivAperture = 1
	winSigma = -1.
	histogramNormType = 0
	L2HysThreshold = 0.2
	gammaCorrection = 1
	nlevels = 64
	signedGradient = True

	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
	#hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

	return hog
	affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR



if __name__ == '__main__':

	print('Loading samples ... ')
	digits, labels = load_samples()

	
	print('Defining HoG parameters ...')
	# HoG feature descriptor
	hog = get_hog();

	print('Calculating HoG descriptor for every image ... ')
	#hog_descriptors = []
	i = 0
	
	for img in digits:

		if(i == 0):
			hog_descriptors = np.array(zip(*hog.compute(img)))
			i = 1
		else:
			hog_descriptors = np.concatenate( ( hog_descriptors,np.array(zip(*hog.compute(img)) ) ) , axis=0)
	
	print('Training SVM model ...')
	model = svmInit()
	model = svmTrain(model, hog_descriptors, labels)

	print('Evaluating model ... ')
	vis = svmEvaluate(model, digits, hog_descriptors, labels)

	cv2.imwrite("digits-classification.jpg",vis)
	cv2.imshow("Vis", vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cap = cv2.VideoCapture('../img/trolley-1.mp4')
	#winW,winH = (75,244)
	winH,winW = (75,244)
	
	while(cap.isOpened()):
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		clone = frame.copy()
		for (x, y, window) in sliding_window(gray, stepSize=10, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			

	 		window = cv2.transpose(window)
			window = cv2.flip(window,flipCode=2)
			cv2.imshow("dasiduahsiduah", window)
			predictions = svmPredict(model, np.array(zip(*hog.compute(window))))

			if(predictions == 1):
				cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			else:
				print(predictions)
			
			cv2.imshow("Window", clone)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			#time.sleep(0.025)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		#time.sleep(0.025)
