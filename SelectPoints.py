# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 22:21:11 2018

@author: vikas upadhyay
"""



import cv2
from pylab import plot, ginput, show, axis
import matplotlib.pyplot as plt
import numpy as np


# matplotlib auto : create seperate plot of image for point selection
def SelectFeaturePoints(img1,name,points):   
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    row,col,depth=img1.shape    
    axis([col,0 ,row,0]) 
    plt.figure(1)   
    imgplot1 = plt.imshow(img1)      
    pts1 = np.round(ginput(points, 900)) # it will wait for eighty clicks    
    np.savetxt(name+'.txt',list(pts1))
    points1=list(np.loadtxt(name+'.txt'))
    print('feature point selection done.....................')
    
    return points1


# interpolated feature points to get intermidate image feature point
def IntermediateImagePoint(points1, points2, alpha):
    MidImgPoints = [];
    # Compute weighted average point coordinates
    for i in range(0, len(points1)):
        x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
        y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
        MidImgPoints.append((x,y))
        
    return MidImgPoints

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
#    size = img.shape
#    r = (0, 0, size[1], size[0])
#    triangleList[:,0]
#    for t in triangleList :
#         
#        pt1 = (t[0], t[1])
#        pt2 = (t[2], t[3])
#        pt3 = (t[4], t[5])
#        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :   
#            # draw line segment
#                cv2.line(img, pt1, pt2, delaunay_color, 1)
#                cv2.line(img, pt2, pt3, delaunay_color, 1)
#                cv2.line(img, pt3, pt1, delaunay_color, 1)
                
    return triangleList

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return dst


# Warps and alpha blends triangular regions from img1 and img2 to interpolated img
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
    







