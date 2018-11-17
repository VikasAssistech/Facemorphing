# CS205 project
# image_analogies_serial.py
# Fangzhou Yu, Jessica Yao

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy.misc

# create kernel weights 
kernel5 = np.array([1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1])/256.

# ANN search function (Approximate nearest neighbor search)
def ANNalgo(flnn,n,hh,ww,feature_vectorA_prime,f_B_reshape):
	# find the corresponding feature vector for the layer
    f_B = np.array(f_B_reshape[n],dtype=np.float32)
    
    hA,wA = feature_vectorA_prime[n].shape[0:2]
    
    	# find the ANN!
    print ("Approximate nearest neighbor takes little time")
    start_time = time.time()
    indices,dist = flnn.knnSearch(f_B,1,params={})
    stop_time = time.time()
    print (" Approximate nearest neighbor done.....................!")
    print ("Time: %f" % (stop_time - start_time))
    
    	# convert the index to row and col
    jminlist = indices % wA
    iminlist = indices/wA
    
    result = np.zeros([hh,ww],dtype=np.float32)
    
    for i in range(hh):
    	for j in range(ww):
    		if i > 2 and j > 2 and i < hh-2 and j < ww-2:
    			k = i*ww+j
    
    imin = iminlist[k,0]; imin=np.int32(imin)
    jmin = jminlist[k,0]
    
    result[i,j] = feature_vectorA_prime[n][imin,jmin][12]
    
    return iminlist,jminlist,result

# brute force search function
def brutealgo(n,i,j,feature_vectorA,feature_vectorA_prime,feature_vectorB):
	# find the corresponding feature vector for i,j
	f_B = feature_vectorB[n][i,j]

	h,w = feature_vectorA[n].shape[0:2]

	minn = float('inf')
	imin = 0
	jmin = 0

	for y in range(h):
		for x in range(w):
			diff = np.linalg.norm(kernel5*(feature_vectorA[n][y,x] - f_B))

			if diff < minn:
				minn = diff
				imin,jmin = y,x

	result = feature_vectorA_prime[n][imin,jmin][12]


	return result

