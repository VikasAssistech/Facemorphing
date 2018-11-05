
"""
Created on Wed Sep 19 22:38:07 2018
@author: vikas upadhyay
ANZ188059
Image morphing using interactive point selecton

"""

import cv2
from pylab import plot, ginput, show, axis
import matplotlib.pyplot as plt
import numpy as np
import time
from SelectPoints import SelectFeaturePoints,IntermediateImagePoint,draw_delaunay, morphTriangle



# main programm    

''' Step 1: read images from local folder '''
img1 = cv2.imread('pulkit.jpg',1)
img2 = cv2.imread('rohan.jpg',1)

row,col,depth=img1.shape 

'''  Step 2: Feature point selection: select the feature points manually 
interactively selecting feature points from left to right and top to bottom in images
| image boundry : 10 points | chin: 12 points  |  Lips : 15 point | left eye: 12 points |
| right eye: 12 points | Head: 12 points | nose : 12 point |
matplotlib auto : will plot images in seperate window for manual selection '''

#fp1 = SelectFeaturePoints(img1, 'pulkit',10)
#fp2 = SelectFeaturePoints(img2, 'rohan' )


''' load interactive feature points selected earlier '''
fp1=list(np.loadtxt('pulkit.txt'))
fp2=list(np.loadtxt('rohan.txt'))

print('Interactive feature points loaded...............')


''' Step 3: find intermidate image feature point for morphing and convert float nparray to tuple '''
fpMid = IntermediateImagePoint(fp1,fp2,0.5)
fp1 = IntermediateImagePoint(fp1,fp1,1)
fp2 = IntermediateImagePoint(fp1,fp2,1)


''' plot selected feature points  '''
plt.figure(1)
#plt.subplot(231)
implot = plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.scatter(x=np.array(fp1)[:,0], y=np.array(fp1)[:,1], c='r',marker='*',s=20)
#
#
#plt.subplot(232)
#implot = plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
#plt.scatter(x=np.array(fp2)[:,0], y=np.array(fp2)[:,1], c='r',marker='*',s=20)

#plt.subplot(133)
#axis([0,col,row,0]) 
#plt.scatter(x=np.array(fpMid)[:,0], y=np.array(fpMid)[:,1], c='r',marker='*',s=20)


''' Define image display window names '''
win_delaunay_p = "Delaunay Triangulation_pulkit"
win_delaunay_r = "Delaunay Triangulation_rohan"


''' Step 4: Draw triangles using delaunay triangulation'''
animate = True
alpha = 0.5
# Define colors for drawing.
delaunay_color = (255,0,255)
# Keep a image copy around
img1_orig = img1.copy(); 
img2_orig = img2.copy(); 
# Rectangle to be used with Subdiv2D
size = img1.shape
rect = (0, 0, size[1], size[0])
# Creates an empty Delaunay subdivision 
subdiv = cv2.Subdiv2D(rect);

## Delaunay triangulation : Inserting points into subdiv, it will return the triangle with min acute angles
for p in fpMid:
    subdiv.insert(p)     
#     Show animation
    if animate :
        img1_copy = img1_orig.copy()
        # Draw delaunay triangles
        draw_delaunay( img1_copy, subdiv, delaunay_color )
#        cv2.imshow(win_delaunay_p, img1_copy)
#        cv2.waitKey(100)
        
# Draw delaunay triangles
tringle1=draw_delaunay( img1, subdiv, delaunay_color)
#cv2.destroyAllWindows()

#
for p in fpMid:
    subdiv.insert(p)     
    # Show animation
    if animate :
        img2_copy = img2_orig.copy()
        # Draw delaunay triangles
        draw_delaunay( img2_copy, subdiv, delaunay_color );
#        cv2.imshow(win_delaunay_r, img3_copy)
#        cv2.waitKey(100)
#
### Draw delaunay triangles
tringle2=draw_delaunay( img2, subdiv, delaunay_color);
#cv2.destroyAllWindows()

tp1=[] 
tp2=[]
np.savetxt('tringle1.txt',tringle1)
np.savetxt('tringle2.txt',tringle2)
#
for i in range(len(tringle1)):
    if np.max(np.abs(tringle1[i,:]))!=1500:
        tp1.append(tringle1[i,:])
    else:
        tp1.append(tringle1[i,:])
#        print(i)

for i in range(len(tringle2)):
    if np.max(np.abs(tringle2[i,:]))!=1500:
        tp2.append(tringle2[i,:])
    else:
        tp2.append(tringle2[i,:])
#        print(i)        
x=[];y=[];t=0

#        
tr=[];trg = np.zeros([len(tp1),3])
for i in range(len(fpMid)):
    for j in range(len(tp1)):
        for k in range(3):
            dist = np.sqrt((tp1[j][2*k]-fpMid[i][0])**2+(tp1[j][2*k+1]-fpMid[i][1])**2)
#            print(dist)
#            dist = np.sqrt((x[j]-fpMid[i][0])**2 + (y[j]-fpMid[i][1])**2)
            if dist==0:
                trg[j,k] = i
                tr.append([i,j,k])
                t=t+1
#                print(i)

np.savetxt('Target_tringle.txt',list(trg))


# Allocate image for final output
MorphImg = np.zeros(img1.shape, dtype = img1.dtype)
trg_n=list(np.loadtxt('Target_tringle.txt'))

# Read triangles from Target_tringle.txt
# display result in fraction of alpha variation

for i in range(21):
    alpha=0.05*i
    for indx in range(len(tp1)) :
        x,y,z = trg_n[indx]
        i=i+1
        x = int(x)
        y = int(y)
        z = int(z)
    # select triangle 
        tp1 = [fp1[x], fp1[y], fp1[z]]
        tp2 = [fp2[x], fp2[y], fp2[z]]
        tMid = [ fpMid[x], fpMid[y], fpMid[z] ]
            # Morph one triangle at a time.
        morphTriangle(img1, img2, MorphImg, tp1, tp2, tMid, alpha)
    cv2.imwrite('res/itration'+str(i)+'.jpg',MorphImg)
# Display Result
    cv2.imshow('display', np.uint8(MorphImg))
    cv2.waitKey(1)
    time.sleep(0.1)

cv2.destroyAllWindows()