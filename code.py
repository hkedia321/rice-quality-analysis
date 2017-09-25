import cv2
import numpy as np
from matplotlib import pyplot as plt
#rnjn
print "hello world"
img = cv2.imread('img3.jpg',0)#load in greyscale mode

#convert into binary
ret,thresh1 = cv2.threshold(img,160,255,cv2.THRESH_BINARY)

#averaging filter
kernel = np.ones((3,3),np.float32)/9
dst = cv2.filter2D(thresh1,-1,kernel)




plt.subplot(1,3,1),plt.imshow(img,'gray')
plt.title("Original image")
plt.subplot(1,3,2),plt.imshow(thresh1,'gray')
plt.title("Binary image")
plt.subplot(1,3,3),plt.imshow(dst,'gray')
plt.title("Filtered image")
plt.show()
