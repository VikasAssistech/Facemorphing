
"""
Created on Wed Sep 12 00:03:48 2018

@author: vikas upadhyay

Image morphing Pulkit over Rohan

"""

import cv2
import numpy as np
import random
from numpy.linalg import inv
import time
# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :    
    # Given a pair of triangles, find the affine transform.
#    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )    
    
#    warpMat, status = cv2.findHomography(np.float32(srcTri), np.float32(dstTri))
#    dst = cv2.warpPerspective(src, warpMat,(width,height))
    
    # alternative affine transform
    pts1 = np.array([[srcTri[0][0],srcTri[1][0],srcTri[2][0]], [srcTri[0][1],srcTri[1][1], srcTri[2][1]],[1,1,1]])
    pts2 = np.array([[dstTri[0][0],dstTri[1][0],dstTri[2][0]], [dstTri[0][1],dstTri[1][1],dstTri[2][1]] ,[1,1,1]])
    if np.linalg.det(pts1)!= 0:
#        print('hi')
        InvA=inv(pts1)
        AffineT=np.matmul(pts2,InvA)
        warpMat = np.float64((AffineT[0,:],AffineT[1,:]))
    else:
#        print('bye')
        warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
  
    
# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
 
# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
    triangleList[:,0]
    for t in triangleList :    
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
         
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)
      
    return triangleList
    

# Define window names
win_delaunay = "Delaunay Triangulation"

#  draw triangles
animate = True
alpha = 0.5

# Define colors for drawing.
delaunay_color = (255,255,255)
points_color = (0, 0, 255) 

# Read in the image.
img1 = cv2.imread("pulkit.jpg");
img2 = cv2.imread("rohan.jpg");

# Keep a copy around
img_orig = img1.copy(); 

# Rectangle to be used with Subdiv2D
size = img1.shape
rect = (0, 0, size[1], size[0])
 
# Create an instance of Subdiv2D
subdiv = cv2.Subdiv2D(rect);
 
# Create an array of points.
points = []; points1_n=[];points2_n=[]     
points1=list(np.loadtxt('pulkit.txt'))
points2=list(np.loadtxt('rohan.txt'))

# Compute weighted average point coordinates
for i in range(0, len(points1)):
    x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
    y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
    points1_n.append((points1[i][0],points1[i][1]))
    points2_n.append((points2[i][0],points2[i][1]))
    points.append((x,y))

np.savetxt('pulkit_rohan.txt',points) 
points=list(np.loadtxt('pulkit_rohan.txt'))

# Insert points into subdiv
#for p in points2_n:       # points pulkit_rohan | points1_n pulkit | points2_n rohan
#    subdiv.insert(p)
#    if animate :
#        img1_copy = img_orig.copy()
#        # Draw delaunay triangles
#        draw_delaunay( img1_copy, subdiv, (255, 255, 255) );
#        cv2.imshow(win_delaunay, img1_copy)
#        cv2.waitKey(100)
### Draw delaunay triangles
#triangle=draw_delaunay( img1, subdiv, (255, 255, 255) );
#
#tr=[]
#for j in range(len(triangle)):
#    tr.append((triangle[j][0],triangle[j][1],triangle[j][2],triangle[j][3],triangle[j][4],triangle[j][5]))
#    
#np.savetxt('triangle_rohan.txt',tr)  

# Show results


# assign feature points to the triangles vertices, total 80 points
#tr_pulkit_n = np.zeros([len(tr_pulkit),3])
#for i in range(len(points1)):
#    for j in range(len(tr_pulkit)):
#        for k in range(3):
#            dist1 = np.sqrt((tr_pulkit[j][2*k]-points1[i][0])**2+(tr_pulkit[j][2*k+1]-points1[i][1])**2)
##            print(dist)
##            dist = np.sqrt((x[j]-fpMid[i][0])**2 + (y[j]-fpMid[i][1])**2)
#            if dist1==0:
#                tr_pulkit_n[j,k] = i
##                tr.append([i,j,k])
#
#np.savetxt('triangle_pulkit.txt',list(tr_pulkit_n))
#
#
#tr_rohan_n = np.zeros([len(tr_rohan),3])
#for i in range(len(points2)):
#    for j in range(len(tr_rohan)):
#        for k in range(3):
#            dist2 = np.sqrt((tr_rohan[j][2*k]-points2[i][0])**2+(tr_rohan[j][2*k+1]-points2[i][1])**2)
##            print(dist)
##            dist = np.sqrt((x[j]-fpMid[i][0])**2 + (y[j]-fpMid[i][1])**2)
#            if dist2==0:
#                tr_rohan_n[j,k] = i
##                tr.append([i,j,k])
#
#np.savetxt('triangle_rohan.txt',list(tr_rohan_n))
#
#tr_pulkit_rohan_n = np.zeros([len(tr_pulkit_rohan),3])
#for i in range(len(points)):
#    for j in range(len(tr_pulkit_rohan)):
#        for k in range(3):
#            dist3 = np.sqrt((tr_pulkit_rohan[j][2*k]-points[i][0])**2+(tr_pulkit_rohan[j][2*k+1]-points[i][1])**2)
##            print(dist)
##            dist = np.sqrt((x[j]-fpMid[i][0])**2 + (y[j]-fpMid[i][1])**2)
#            if dist3==0:
#                tr_pulkit_rohan_n[j,k] = i
##                tr.append([i,j,k])
#
#np.savetxt('triangle_pulkit_rohan.txt',list(tr_pulkit_rohan_n))

tr_pulkit=list(np.loadtxt('triangle_pulkit.txt'))
tr_rohan= list(np.loadtxt('triangle_rohan.txt'))
tr_pulkit_rohan= list(np.loadtxt('triangle_pulkit_rohan.txt'))

# Allocate space for final output
img1Morph = np.zeros(img1.shape, dtype = img1.dtype)
i=0
interFrame=20
for i in range(interFrame+1):
    alpha=i*1/interFrame
    for indx in range(len(tr_pulkit)) :
        x,y,z = tr_pulkit[indx]
        i=i+1
        x = int(x)
        y = int(y)
        z = int(z)
    #    print(i)
        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [ points[x], points[y], points[z] ]
            # Morph one triangle at a time.
        morphTriangle(img1, img2, img1Morph, t1, t2, t, alpha)
    cv2.imwrite('res/itration'+str(i)+'.jpg',img1Morph)
# Display Result
    cv2.imshow('display', np.uint8(img1Morph))
    time.sleep(0.1)
    cv2.waitKey(1)

cv2.destroyAllWindows()


