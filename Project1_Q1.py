#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:34:06 2022

@author: mothish
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

def get_frame():
    vidcap = cv2.VideoCapture('1tagvideo.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        if(count==3):
            break
    

def getMarkerCorners(img,point_size):
    corners = cv2.goodFeaturesToTrack(img, point_size, 0.01, 10)
    corners = np.int0(corners)
    sorted_corners=[]
   
    
    for i in corners:   
       x, y = i.ravel()
       coord=[x,y]
       sorted_corners.append(coord)
     
     
    sorted_corners= np.asarray(sorted_corners)
    sorted_corners_size=sorted_corners.shape[0]
    
    rectangle_corner= []
    #sorting according to x coordinate
    sorted_corners_x = sorted_corners[sorted_corners[:, 0].argsort()]
    
    rectangle_corner.append(sorted_corners_x[0])
    rectangle_corner.append(sorted_corners_x[1])
    rectangle_corner.append(sorted_corners_x[sorted_corners_size-1])
    rectangle_corner.append(sorted_corners_x[sorted_corners_size-2])


    #sorting according to y coordinate
    #sorted_corners_y = sorted_corners[sorted_corners[:, 1].argsort()]
    
    


    rectangle_corner= np.asarray(rectangle_corner)
    return rectangle_corner   

        
def get_FFt_Image(img):
    img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ( _ , img) = cv2.threshold(img,200, 255, cv2.THRESH_BINARY)
    fft = fftpack.fft2(img,axes=(0,1))
    fft_shift= fftpack.fftshift(fft)
 
    
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    
    mask = np.ones((rows, cols), np.uint8)
    r = 80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    fshift = fft_shift * mask


    f_ishift = fftpack.ifftshift(fshift)
    img_back = fftpack.ifft2(f_ishift)
    img_back=np.abs(img_back)
    ( _ , bin_image) = cv2.threshold(img_back, 50, 255, cv2.THRESH_BINARY)
    return bin_image
    

def getOrientationOfCorners(coord):

    coords_sorted_x = coord[np.argsort(coord[:,0])]

    coords_left = coords_sorted_x[0:2, :]
    
    coords_left_sorted = coords_left[np.argsort(coords_left[:,1])]
    top_left, botttom_left = coords_left_sorted

    
    coords_right = coords_sorted_x[2:4, :]

    coords_right_sorted= coords_right[np.argsort(coords_right[:,1])]
    top_right, bottom_right = coords_right_sorted



    coords_sorted = np.array([top_left, botttom_left, bottom_right, top_right])
    return coords_sorted
    
def computeHomographyfor2Points(marker_corner, warping_corners):

    if (len(marker_corner) < 4) or (len(warping_corners) < 4):
        print("Need atleast four points to compute SVD.")
        return 0

    x_marker = marker_corner[:, 0]
    y_marker = marker_corner[:, 1]
    x_warping = warping_corners[:, 0]
    y_warping = warping_corners[:,1]

    A = []
    for i in range(4):
        r1 = np.array([-x_marker[i], -y_marker[i], -1, 0, 0, 0, x_marker[i]*x_warping[i], y_marker[i]*x_warping[i], x_warping[i]])
        r2 = np.array([0, 0, 0, -x_marker[i], -y_marker[i], -1, x_marker[i]*y_warping[i], y_marker[i]*y_warping[i], y_warping[i]])
        A.append(r1)
        A.append(r2)
        
        

    A = np.array(A)
    U, E, V = np.linalg.svd(A)
    V = V.transpose()
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    X=H[2,2]
    if(H[2,2]==0):
        X=1    
    H = H / X
    
    return H

             
    
def applyHomography2ImageUsingInverseWarping(image, H, size):

    Yt, Xt = np.indices((size[0], size[1]))
    lin_homg_pts_trans = np.stack((Xt.ravel(), Yt.ravel(), np.ones(Xt.size)))

    H_inv = np.linalg.inv(H)
    lin_homg_pts = H_inv.dot(lin_homg_pts_trans)
    lin_homg_pts /= lin_homg_pts[2,:]

    Xi, Yi = lin_homg_pts[:2,:].astype(int)
    Xi[Xi >=  image.shape[1]] = image.shape[1]
    Xi[Xi < 0] = 0
    Yi[Yi >=  image.shape[0]] = image.shape[0]
    Yi[Yi < 0] = 0

    image_transformed = np.zeros((size[0], size[1], 3))
    image_transformed[Yt.ravel(), Xt.ravel(), :] = image[Yi, Xi, :]
    
    return image_transformed


def extractInfoFromTag(tag):
    tag_size = tag.shape[0]
    grid_size = 8
    pixels_in_one_grid =  int(tag_size/8)

    info_with_padding = np.zeros((8,8))

    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            grid = tag[i*pixels_in_one_grid:(i+1)*pixels_in_one_grid, j*pixels_in_one_grid:(j+1)*pixels_in_one_grid]
            
            if np.sum(grid) > 100000*0.7 and np.median(grid) == 255:
                # print(np.sum(grid))
                info_with_padding[i,j] = 255
    # print(info_with_padding)
    info = info_with_padding[2:6, 2:6]
    return info
def InverseWarping(image, H, x,y):

    Y,X =np.indices((y,x))
    
    h_t = np.stack((X.ravel(), Y.ravel(), np.ones(X.size)))

    H_inv = np.linalg.inv(H)
    h_p = H_inv.dot(h_t)
    h_p /= h_p[2,:]

    X_h, Y_h = h_p[:2,:].astype(int)
    
    Y_h[Y_h >=  image.shape[0]] = image.shape[0]
    Y_h[Y_h < 0] = 0
    
    X_h[X_h >=  image.shape[1]] = image.shape[1]
    X_h[X_h < 0] = 0
   
            
    inverse_warping= np.zeros((x, y, 3))
    inverse_warping[Y.ravel(), X.ravel(), :] = image[Y_h, X_h, :]
    
    return inverse_warping
   

def getMarkerOrientation(marker):
    marker_size = marker.shape[0]
    grid_size = 8
    pixels_in_one_grid =  int(marker_size/8)

    info_with_padding = np.zeros((8,8))

    for i in range(grid_size):
        for j in range(grid_size):
            grid = marker[i*pixels_in_one_grid:(i+1)*pixels_in_one_grid, j*pixels_in_one_grid:(j+1)*pixels_in_one_grid]
            
            if np.sum(grid) > 100000*0.7 and np.median(grid) == 255:
                #print(np.sum(grid))
                info_with_padding[i,j] = 255
    # print(info_with_padding)
    info = info_with_padding[2:6, 2:6]
    return info



def getTagID(info):
    while not info[3,3]:
        info = np.rot90(info, 1)

    # print(info)
    id_info = info[1:3, 1:3]
    id_info_flat = np.array([id_info[0,0], id_info[0,1], id_info[1,1], id_info[1,0]])
    tag_id = 0
    
    for i in range(4):
        if(id_info_flat[i]):
            tag_id = tag_id + 2**(i)

    
    return tag_id



if __name__ == "__main__":
    #get_frame()
    img = cv2.imread("tag1.png")
    fft_image=get_FFt_Image(img)
    #print(fft_image.shape)
    fft_image = cv2.GaussianBlur(fft_image,(11,11),cv2.BORDER_DEFAULT)
    fft_image=np.float32(fft_image)
    


    paper = cv2.GaussianBlur(fft_image,(11,11),cv2.BORDER_DEFAULT) 
    marker_corner = getMarkerCorners(paper,12)
     
    
    marker_corner= np.float32(getOrientationOfCorners(marker_corner))
    
    markerOrientation = 160
    warping_corners = np.float32(getOrientationOfCorners(np.array([ [0, markerOrientation], [markerOrientation, markerOrientation], [markerOrientation, 0], [0, 0]])))
    
    Htd = computeHomographyfor2Points(marker_corner,warping_corners)
    marker = InverseWarping(img, Htd, markerOrientation, markerOrientation)
    marker = cv2.cvtColor(np.uint8(marker), cv2.COLOR_BGR2GRAY)
    
    marker_data = getMarkerOrientation(marker)
    
    
    tag_id = getTagID(marker_data)
    
    print(tag_id)
    
    #print(tag_corner)
    
    
    


    

    
   # for i in range(4):
    #  x=marker_corner[i][0]
     # y= marker_corner[i][1]
      #cv2.circle(img, (x, y), 3, (0,255,0), -1)
     

# =============================================================================
#     fig= plt.figure(figsize=(12,12))
#     ax1= fig.add_subplot(2,2,1)
#     ax1.imshow(img)
#     ax1.title.set_text('Input Image')
#     ax2= fig.add_subplot(2,2,2)
#     ax2.imshow(magnitude_spectrum)
#     ax2.title.set_text('Magnitude_spectrum Image') 
#     ax3 = fig.add_subplot(2,2,3)
#     ax3.imshow(fshift_mask_mag, cmap='gray')
#     ax3.title.set_text('FFT + Mask')
#     ax4 = fig.add_subplot(2,2,4)
#     ax4.imshow(img_back, cmap='gray')
#     ax4.title.set_text('After inverse FFT')
#     plt.show()
# 
# =============================================================================
    plt.imshow(marker,cmap='gray')
    print("Code")
