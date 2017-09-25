import cv2
import numpy as np
from matplotlib import pyplot as plt
#rnjn
print "Starting"
img = cv2.imread('rice.png',0)#load in greyscale mode

#convert into binary
ret,binary = cv2.threshold(img,160,255,cv2.THRESH_BINARY)# 160 - threshold, 255 - value to assign, THRESH_BINARY_INV - Inverse binary

#averaging filter
kernel = np.ones((5,5),np.float32)/9
dst = cv2.filter2D(binary,-1,kernel)# -1 : depth of the destination image


kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

#erosion
erosion = cv2.erode(dst,kernel2,iterations = 1)

#dilation 
dilation = cv2.dilate(erosion,kernel2,iterations = 1)

#edge detection
edges = cv2.Canny(dilation,100,200)

#plot the images
imgs_row=2
imgs_col=3
plt.subplot(imgs_row,imgs_col,1),plt.imshow(img,'gray')
plt.title("Original image")

plt.subplot(imgs_row,imgs_col,2),plt.imshow(binary,'gray')
plt.title("Binary image")

plt.subplot(imgs_row,imgs_col,3),plt.imshow(dst,'gray')
plt.title("Filtered image")

plt.subplot(imgs_row,imgs_col,4),plt.imshow(erosion,'gray')
plt.title("Eroded image")

plt.subplot(imgs_row,imgs_col,5),plt.imshow(dilation,'gray')
plt.title("Dialated image")

plt.subplot(imgs_row,imgs_col,6),plt.imshow(edges,'gray')
plt.title("Edge detect")

plt.show()
