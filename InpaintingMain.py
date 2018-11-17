"""
Created on Thu Oct 18 20:19:12 2018

@author: Vikas Upadhyay (ANZ188059)
Digital Image analysis Image inpainting using Fast Marching Method

"""

import InpaintFMM
import cv2
import numpy as np
from InpaintFMM import*

img = cv2.imread('input/lena_in.png')
#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_original = np.copy(img)
mask_img = cv2.imread('input/lena_mask.png')
mask = mask_img[:, :, 0].astype(bool, copy=False)
inpaint(img, mask)
cv2.imwrite('output/lena_output.jpg', img)


cv2.imshow('Original',np.uint8(img_original))
cv2.imshow('Inpainted',np.uint8(img))
cv2.waitKey(0)

cv2.destroyAllWindows()

