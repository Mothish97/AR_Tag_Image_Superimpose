# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 22:48:29 2022

@author: mothi
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def get_frame():
    vidcap = cv2.VideoCapture('1tagvideo.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        if(count==400):
            break
    

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
    


def getCorners(img,point_size):
    corners = cv2.goodFeaturesToTrack(img, point_size, 0.01, 10)
    corners = np.int0(corners)
    sorted_corners=[]
   
    
    for i in corners:   
       x, y = i.ravel()
       coord=[x,y]
       sorted_corners.append(coord)
     
     
    sorted_corners= np.asarray(sorted_corners)
    sorted_corners_size=sorted_corners.shape[0]
    all_corner=sorted_corners
    rectangle_corner= []
    
    #sorting according to x coordinate
    sorted_corners_x = sorted_corners[sorted_corners[:, 0].argsort()]
    rectangle_corner.append(sorted_corners_x[0])
    rectangle_corner.append(sorted_corners_x[sorted_corners_size-1])


    #sorting according to y coordinate
    sorted_corners_y = sorted_corners[sorted_corners[:, 1].argsort()]
    rectangle_corner.append(sorted_corners_y[0])
    rectangle_corner.append(sorted_corners_y[sorted_corners_size-1])
    

    rectangle_corner= np.asarray(rectangle_corner)
    
    
    
    #All Corners for marker
    corners = cv2.goodFeaturesToTrack(img, 25, 0.01, 10)
    corners = np.int0(corners)
    sorted_corners=[]
   
    
    for i in corners:   
       x, y = i.ravel()
       coord=[x,y]
       sorted_corners.append(coord)
     
    sorted_corners= np.asarray(sorted_corners)
    
    
    return rectangle_corner,all_corner  

def getMarkerCorners(paper_corner,all_corner):
    
    paper_corner = getOrientationOfCorners(paper_corner)
    
    marker_points=[]
    
    for i in range(len(all_corner)):
        point = Point(all_corner[i][0], all_corner[i][1])
        polygon = Polygon([paper_corner[1],paper_corner[2],paper_corner[3], paper_corner[0]])
        
        # for j in range(len(paper_corner)):
        #     if(abs(all_corner[i][0]-paper_corner[j][0])< 4 or abs(all_corner[i][1]-paper_corner[j][1])<10):
        #         all_corner[i][0]= paper_corner[j][0]
        #         all_corner[i][1]= paper_corner[j][1]
        #         point = Point(all_corner[i][0], all_corner[i][1])        
        
        if polygon.contains(point):
            marker_points.append(all_corner[i])
    
    marker_points=np.asarray(marker_points) 
    
    
    marker_points= np.asarray(marker_points)
    marker_points_size=marker_points.shape[0]
    
    rectangle_corner= []
    #sorting according to x coordinate
    marker_points_x = marker_points[marker_points[:, 0].argsort()]
    
    rectangle_corner.append(marker_points_x[0])
    rectangle_corner.append(marker_points_x[marker_points_size-1])


    #sorting according to y coordinate
    marker_points_y = marker_points[marker_points[:, 1].argsort()]
    rectangle_corner.append(marker_points_y[0])
    rectangle_corner.append(marker_points_y[marker_points_size-1])            
    
    rectangle_corner= np.asarray(rectangle_corner)
    return rectangle_corner  
            
def getCornersfor3Points(img,point_size):
    corners = cv2.goodFeaturesToTrack(img, point_size, 0.01, 10)
    corners = np.int0(corners)
    sorted_corners=[]
   
    
    for i in corners:   
       x, y = i.ravel()
       coord=[x,y]
       sorted_corners.append(coord)
     
     
    sorted_corners= np.asarray(sorted_corners)
    sorted_corners_size=sorted_corners.shape[0]
    all_corner=sorted_corners
    
    rectangle_corner= []
    #sorting according to x coordinate
    sorted_corners_x = sorted_corners[sorted_corners[:, 0].argsort()]
    
    rectangle_corner.append(sorted_corners_x[0])
    rectangle_corner.append(sorted_corners_x[sorted_corners_size-1])
    
    
    sorted_corners_x = sorted_corners_x[1:sorted_corners_size-1]
    rectangle_corner.append(sorted_corners_x[0])
    rectangle_corner.append(sorted_corners_x[sorted_corners_size-3])
    
    rectangle_corner= np.asarray(rectangle_corner)
    
    #All Corners for marker
    corners = cv2.goodFeaturesToTrack(img, 25, 0.01, 10)
    corners = np.int0(corners)
    sorted_corners=[]
   
    
    for i in corners:   
       x, y = i.ravel()
       coord=[x,y]
       sorted_corners.append(coord)
     
    sorted_corners= np.asarray(sorted_corners)
    
    
    return rectangle_corner,all_corner  



def getMarkerCornersfor3Points(paper_corner,all_corner):
    
    paper_corner = getOrientationOfCorners(paper_corner)
    
    marker_points=[]
    
    for i in range(len(all_corner)):
        point = Point(all_corner[i][0], all_corner[i][1])
        polygon = Polygon([paper_corner[1],paper_corner[2],paper_corner[3], paper_corner[0]])
        if polygon.contains(point):
            marker_points.append(all_corner[i])
    
    marker_points=np.asarray(marker_points) 
    
    
    marker_points= np.asarray(marker_points)
    marker_points_size=marker_points.shape[0]
    
    rectangle_corner= []
    #sorting according to x coordinate
    marker_points_x = marker_points[marker_points[:, 0].argsort()]
    
    rectangle_corner.append(marker_points_x[0])
    rectangle_corner.append(marker_points_x[marker_points_size-1])

    marker_points_x = marker_points_x[1:marker_points_size-1]
    rectangle_corner.append(marker_points_x[0])
    rectangle_corner.append(marker_points_x[marker_points_size-3])
    rectangle_corner= np.asarray(rectangle_corner) 
    
    return rectangle_corner  
            
    
def InverseWarping(image, H, x,y):

    Y,X =np.indices((y,x))
    
    h_t = np.stack((X.ravel(), Y.ravel(), np.ones(X.size)))

    H_inv = np.linalg.inv(H)
    h_p = H_inv.dot(h_t)
    h_p /= h_p[2,:]

    X_h, Y_h = h_p[:2,:].astype(int)
    
    Y_h[Y_h >=  image.shape[0]] = image.shape[0]-1
    Y_h[Y_h < 0] = 0
    
    X_h[X_h >=  image.shape[1]] = image.shape[1]-1
    X_h[X_h < 0] = 0
   
    
    for i in range(len(Y_h)):
        if(Y_h[i]>=1080):
            Y_h[i]=1079
        if(Y_h[i]<0):
            Y_h[i]=0
                 
    for i in range(len(X_h)):
        if(X_h[i]>=1080):
            X_h[i]=1079
        if(X_h[i]<0):
            X_h[i]=0
            
    inverse_warping= np.zeros((x, y, 3))
    inverse_warping[Y.ravel(), X.ravel(), :] = image[Y_h, X_h, :]
    
    return inverse_warping
    

def putHomography(coords, H):
    Xi = coords[:, 0]
    Yi = coords[:, 1]
    h_p = np.stack((Xi, Yi, np.ones(Xi.size)))
    

    ht = H.dot(h_p)
    
    for i in range  (len(ht[2,:])):
        if(ht[2,i]==0):
            ht[2,i]=1    
            
    ht /= ht[2,:]

    X, Y = ht[:2,:].astype(int)
    coords = np.dstack([X, Y])
    return coords

def rotateCoords(coords):

   coords_copy = list(coords.copy())
   coords_new = coords_copy.pop(-1)
   coords_copy.insert(0, coords_new)
   return np.array(coords_copy) 

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

    
def get_FFt_Image(img):
    img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ( _ , img) = cv2.threshold(img,190, 255, cv2.THRESH_BINARY)
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
    

def getProjection(H, Cam):
    Cam_inv = np.linalg.inv(Cam)

    B_t = np.dot(Cam_inv, H)
    B_t_m = np.linalg.norm(B_t)
    if B_t_m < 0:
        B = -1  * B_t
    else:
        B =  B_t

    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]

    l = (np.linalg.norm(b1) + np.linalg.norm(b2))/2
    l = 1 / l

    c1 = l * b1
    c2 = l * b2
    c3 = np.cross(c1, c2)
    t = l* b3

    P = np.array([c1,c2, c3, t]).T
    P = np.dot(Cam, P)
    X=P[2,3]
    
    if(P[2,3]<=0):
        X=1
    P = P / X
    return P

def drawCube(image, bottom_points, top_points):

    cv2.drawContours(image, [bottom_points], 0, (0, 255 ,255),3)
    cv2.drawContours(image, [top_points], 0, (0, 255, 255),3)

    for i in range(0, bottom_points.shape[0]):
        color = (0, int(255/(i+1)), int(255/(i+1)))
        cv2.line(image, (bottom_points[i,0], bottom_points[i,1]), (top_points[i,0], top_points[i,1]), color, 3)

    return image

def putProjection(coords, P):
    Xh = coords[:, 0]
    Yh = coords[:, 1]
    Zh = coords[:, 2]

    H_p = np.stack((Xh, Yh, Zh, np.ones(Xh.size)))
    H_t = P.dot(H_p)
    H_t /= H_t[2,:]
    X = H_t[0,:].astype(int)
    Y = H_t[1,:].astype(int)

    projected_points = np.dstack((X,Y)).reshape(4,2)
    return projected_points



def getMarker(marker_corner,img):
    
    marker_corner= np.float32(getOrientationOfCorners(marker_corner))
    
    tag_size = 160
    warping_corners = np.float32(getOrientationOfCorners(np.array([ [0, tag_size], [tag_size, tag_size], [tag_size, 0], [0, 0]])))
    
    Htd = computeHomographyfor2Points(marker_corner,warping_corners)
    marker = InverseWarping(img, Htd, tag_size, tag_size)
    marker = cv2.cvtColor(np.uint8(marker), cv2.COLOR_BGR2GRAY)
    ( _ , marker) = cv2.threshold(marker, 200, 255, cv2.THRESH_BINARY)
    return marker



if __name__ == "__main__":
    
    vidcap = cv2.VideoCapture('1tagvideo.mp4')
    count = 0
    cam = np.array([[1346.10059534175, 0, 0], [0, 1355.93313621175, 0], [932.163397529403, 654.898679624155, 1]])
    cam = np.transpose(cam)
    
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    size=(frame_width,frame_height)
    
    result = cv2.VideoWriter('CubeSuperImposed.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)
    while True:
        success,frame = vidcap.read()
        if not success:
            print("Stream ended..")
            break
        
        img = frame
        fft_image=get_FFt_Image(img)
        fft_image = cv2.GaussianBlur(fft_image,(11,11),cv2.BORDER_DEFAULT)
        fft_image=np.float32(fft_image)
        
    
        paper = cv2.GaussianBlur(fft_image,(11,11),cv2.BORDER_DEFAULT)
        paper_corner,all_corner = getCorners(paper,28)
        isRepeated=False
        
        for i in range(len(paper_corner)):
            for j in range(len(paper_corner)):
                if(paper_corner[i][0]== paper_corner[j][0]  and paper_corner[i][1]== paper_corner[j][1] and i!=j):
                    isRepeated=True
                    #print("Repeated")
                    break
        
        if isRepeated:
            paper_corner,all_corner = getCornersfor3Points(paper,28)
            marker_corner= getMarkerCornersfor3Points(paper_corner,all_corner)
        else:
            marker_corner= getMarkerCorners(paper_corner,all_corner)
        

        
        marker= getMarker(marker_corner,img)
    
        
        marker_corner = getOrientationOfCorners(marker_corner)
        
        tag_info = getMarkerOrientation(marker)
        
        
        ARcorners = np.array([tag_info[0,0], tag_info[0,3], tag_info[3,0], tag_info[3,3]])
        first_time=True
        rotation = 0 
        rows,cols,ch = img.shape 
        image_show = img.copy()
        
        if np.sum(ARcorners) == 255:
            while tag_info[3,3]==0:
                tag_info = np.rot90(tag_info, 1)
                rotation = rotation + 1
                
            num_rotations = int(rotation)
            for n in range(num_rotations):
                marker_corner = rotateCoords(marker_corner)
    
        
        
        
        marker_orientation = 160
        warping_corners = np.float32(getOrientationOfCorners(np.array([ [0, marker_orientation], [marker_orientation, marker_orientation], [marker_orientation, 0], [0, 0]])))
    
        cube_height = np.array([-(marker_orientation), -(marker_orientation), -(marker_orientation), -(marker_orientation)]).reshape(-1,1)
        cube_corners = np.concatenate((warping_corners, cube_height), axis = 1)    
        
        H = computeHomographyfor2Points(np.float32(warping_corners), np.float32(marker_corner))
        Hnew = getProjection(H, cam)
        marker_corner_t = putProjection(cube_corners, Hnew)        
        
        
        count=count+1
        print(count)
        image_cube = drawCube(img, marker_corner, marker_corner_t)
        result.write(image_cube)
    
        cv2.imshow('frame', np.uint8(image_cube))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            

    vidcap.release()
    result.release()
    cv2.destroyAllWindows() 

    


    

    
    for i in range(len(all_corner)):
      x=all_corner[i][0]
      y= all_corner[i][1]
      cv2.circle(img, (x, y),7, (0,0,255), -1)
     
    plt.imshow(img,cmap='gray')
    print("Code")
