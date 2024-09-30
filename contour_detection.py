#contour detection
import cv2 as cv #opencv
import matplotlib.pyplot as plt #matplot to display the result

img = cv.imread('img3.jpg',cv.IMREAD_GRAYSCALE) #for contour detection, we need binary images, and for that we need first grayscale

#for better accuracy, as mentioned before, we need binary images so either canny edge detection/threshold
#here we will use threshold
ret, thresh = cv.threshold(img, 127, 255, 0) #applies thresholding and conv to binary image, ret is the threshold value that we give in
#contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #retr_tree will return all possible contours from image
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #retr_external will return only the external contours, and eliminates internal contours.
plt.figure()
#orig image
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title('Original Image')
plt.axis('off')

#edges
plt.subplot(1,2,2)
img_copy =  cv.cvtColor(img, cv.COLOR_GRAY2BGR) #conv grayscale image to 3-channel color before drawing the contours
cv.drawContours(img_copy,contours,-1,(0,255,0),3)
image_rgb = cv.cvtColor(img_copy, cv.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title('Contours')
plt.axis('off')

plt.show()
