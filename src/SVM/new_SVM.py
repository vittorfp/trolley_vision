#!/usr/bin/env python

import cv2
import time
import numpy as np

SZ = 20
wdt = 140
hei = 300

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
	fh = open("labels.txt", "r") 
	labels = fh.read()
	labels = np.array( labels.split('\n')[:-1] ).astype(int)
	print(labels)
	fh.close()
	n_samples = len(labels)

	positive = labels == 1
	
	negative = labels == 0
	
	#labels = []
	#j = 0
	#for i in range(n_samples):
	#	if(positive[i] == True):
	#		samples_img.append(cv2.imread('../img/dataset2/img'+ str(i) + '.jpg', 0))
	#		j += 1
	#		labels.append(1)
	#n_positive = len(labels)

	#for i in range(n_samples):
	#	if(positive[i] == False) & ( j - n_positive < n_positive):
	#		samples_img.append(cv2.imread('../img/dataset2/img'+ str(i ) + '.jpg', 0))
	#		j += 1
	for i in range(n_samples):
		samples_img.append(cv2.imread('../img/dataset2/img'+ str(i ) + '.jpg', 0))
	print(len(samples_img)	)
	print(len(labels)	)
	return samples_img, np.array(labels)


def svmInit(C=12.5, gamma=0.50625):
	# Set up SVM for OpenCV 3
	svm = cv2.ml.SVM_create()
	# Set SVM type
	svm.setType(cv2.ml.SVM_C_SVC)
	# Set SVM Kernel to Radial Basis Function (RBF) 
	svm.setKernel(cv2.ml.SVM_LINEAR)
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
	print(predictions)
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
	winSize = (140,300)
	blockSize = (40,40)
	blockStride = (20,20)
	cellSize = (20,20)
	nbins = 19
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

def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*140*skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (140, 300), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img

if __name__ == '__main__':

	print('Loading samples ... ')
	digits, labels = load_samples()

	print('Deskew images ... ')
	digits_deskewed = list(map(deskew, digits))
		
	print('Defining HoG parameters ...')
	# HoG feature descriptor
	hog = get_hog();

	print('Calculating HoG descriptor for every image ... ')
	hog_descriptors = []
	i = 0
	for img in digits:
	
		if(i == 0):
			hog_descriptors = np.array(zip(*hog.compute(img)))
			i = 1
		else:
			hog_descriptors = np.concatenate( ( hog_descriptors,np.array(zip(*hog.compute(img)) ) ) , axis=0)
			i = i+1
			print(i)

	hog_descriptors = []
	for img in digits:
		hog_descriptors.append(hog.compute(img))
	hog_descriptors = np.squeeze(hog_descriptors)

	print('Spliting data into training (90%) and test set (10%)... ')
	#train_n = np.random.randint( len(hog_descriptors), size=int(0.9*len(hog_descriptors)) ).astype(int)
	#digits_train, digits_test = digits[train_n]
	#hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
	#labels_train, labels_test = np.split(labels, [train_n])
	
	print('Training SVM model ...')
	
	#model = svmInit()
	model = cv2.ml.SVM_create()
	model.trainAuto( hog_descriptors, cv2.ml.ROW_SAMPLE, labels)
	model = svmTrain(model, hog_descriptors, labels)

	#model = cv2.ml.SVM_create()
	#model.load("../classifier/svm.xml")
	
	print('Evaluating model ... ')
	vis = svmEvaluate(model, digits, hog_descriptors, labels)

	cv2.imwrite("digits-classification.jpg",vis)
	cv2.imshow("Vis", vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cap = cv2.VideoCapture('../img/lenta.mp4')
	#winW,winH = (75,244)
	winH,winW = (140,300)
	
	while(cap.isOpened()):
		ret, frame = cap.read()
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#[0:300,200:340]
		bias = (0,200)
		clone = frame.copy()
		for (x, y, window) in sliding_window(gray, stepSize=10, windowSize=(winH, winW)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winW or window.shape[1] != winH:
				continue

			

			#window = cv2.transpose(window)
			#window = cv2.flip(window,flipCode=2)
			cv2.imshow("Janela deslizante", window)
			#hog_descriptors = []
			#hog_descriptors = hog_descriptors.append(np.array(zip(*hog.compute(window))))
			hog_descriptors = np.array([])
			hog_descriptors = np.concatenate( ( np.array(zip(*hog.compute(img))), np.array(zip(*hog.compute(img)))  ) , axis=0 )
			#print(hog_descriptors.shape)
			#print(hog.compute(img))
			#mat = np.squeeze(hog.compute(img))
			#print(mat)
			predictions = svmPredict(model,  hog_descriptors )
			#sv.predict(samples)
			if(predictions[0] == 0.):
				cv2.rectangle(clone, (x+200, y), (x + winH+200, y + winW), (0, 255, 0), 2)
			
			print(predictions[0])
			
			cv2.imshow("Window", clone)
			if cv2.waitKey(1) & 0xFF == ord('p'):
					break
			
			#time.sleep(0.025)
		
		if cv2.waitKey(0) & 0xFF == ord('q'):
			break
		#time.sleep(0.025)
