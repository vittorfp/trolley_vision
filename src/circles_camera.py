import pypylon
import numpy as np
import cv2

print('\n\nBuild against pylon library version:', pypylon.pylon_version.version)

available_cameras = pypylon.factory.find_devices()
print('Available cameras are', available_cameras)

# Grep the first one and create a camera for it
cam = pypylon.factory.create_device(available_cameras[-1])

# We can still get information of the camera back
print('Camera info :', cam.device_info)

# Open camera and grep some images
cam.open()

def proc_image(image):

	output = image.copy()
	gray = image


	#gray = cv2.GaussianBlur(gray,(5,5),0)

	# detect circles in the image
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.4,140, maxRadius = 100)

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

	return output


while True:
	for image in cam.grab_images(1):
		#print(image.shape)
		result = proc_image(image)
		cv2.imshow('frame',result)
		
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	#cv2.destroyAllWindows()

cv2.destroyAllWindows()