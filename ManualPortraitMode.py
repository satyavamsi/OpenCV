import numpy as np
from matplotlib import pyplot as plt
import cv2

#load your image here
img = cv2.imread('/home/vamsi/test.jpg')
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#rough coordinates of your object in image
start_x = 510
start_y = 19
width = 1500
height = 1300

rect = (start_x,start_y,width,height)

# grabcut algorithm for foreground extraction
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
