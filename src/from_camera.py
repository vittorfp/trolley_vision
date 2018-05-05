import numpy as np
import argparse
import time
import cv2

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True, help = "Path to the image")
#ap.add_argument("-s", "--save", help = "Image to save")
#args = vars(ap.parse_args())


# load the image, clone it for output, and then convert it to grayscale
#image = cv2.imread(args["image"])
cap = cv2.VideoCapture('../img/lenta.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray[200:500,:]
    output = frame.copy()
    
    gray = cv2.GaussianBlur(gray,(3,3),0)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2,40, maxRadius = 150)
    
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imshow('frame',frame[0:300,200:340])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #time.sleep(0.1)


