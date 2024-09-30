#canny edge detection
import cv2 as cv #opencv
import matplotlib.pyplot as plt #matplot to display the result

img = cv.imread('img1.jpg',cv.IMREAD_GRAYSCALE) #for canny edge detection, the image must be in grayscale
edges = cv.Canny(img,100,200,L2gradient=True) #apply canny edge detection algo, which will do the 5 steps mentioned and return the edges
#not entering aperture size, as 3 is the default size, which is what we want
plt.figure()
#orig image
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title('Original Image')
plt.axis('off')

#edges
plt.subplot(1,2,2)
plt.imshow(edges,cmap='gray')
plt.title('Edges')
plt.axis('off')

plt.show()
