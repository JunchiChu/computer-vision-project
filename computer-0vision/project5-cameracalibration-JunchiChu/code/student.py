# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech

import numpy as np
import random
from random import sample

# Returns the projection matrix for a given set of corresponding 2D and
# 3D points. 
# 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
# 'Points_3D' is nx3 matrix of 3D coordinate of points in the world
# 'M' is the 3x4 projection matrix
def calculate_projection_matrix(Points_2D, Points_3D):
    # To solve for the projection matrix. You need to set up a system of
    # equations using the corresponding 2D and 3D points:
    #
    #                                                     [M11       [ u1
    #                                                      M12         v1
    #                                                      M13         .
    #                                                      M14         .
    #[ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
    #  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
    #  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
    #  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
    #                                                      M32         un
    #                                                      M33         vn ]
    #
    # Then you can solve this using least squares with the 'np.linalg.lstsq' operator.
    # Notice you obtain 2 equations for each corresponding 2D and 3D point
    # pair. To solve this, you need at least 6 point pairs. Note that we set
    # M34 = 1 in this scenario. If you instead choose to use SVD via np.linalg.svd, you should
    # not make this assumption and set up your matrices by following the 
    # set of equations on the project page. 
    #
    ##################
    # Your code here #
    ##################

    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    # Your total residual should be less than 1.
    print('Randomly setting matrix entries as a placeholder')
    M = np.array([[-0.4583, 0.2947, 0.0139, -0.0040],
                  [0.0509, 0.0546, 0.5410, 0.0524],
                  [-0.1090, -0.1784, 0.0443, -0.5968]])
    b=np.ones((Points_2D.size,1))
    A=np.ones((Points_2D.shape[0]*2,11))
    index_b=0
    for e in range(Points_2D.shape[0]):
        for m in range(2):
            b[index_b]=Points_2D[e][m]
            index_b+=1
    i=0
    for x in range(0,2*Points_2D.shape[0],2):
        A[x]=[Points_3D[i][0],Points_3D[i][1],Points_3D[i][2],1,0,0,0,0,-Points_3D[i][0]*b[x],-Points_3D[i][1]*b[x],-Points_3D[i][2]*b[x]]
        A[x+1]=[0,0,0,0,Points_3D[i][0],Points_3D[i][1],Points_3D[i][2],1,-Points_3D[i][0]*b[x+1],-Points_3D[i][1]*b[x+1],-Points_3D[i][2]*b[x+1]]
        i+=1
    #print(b)
    #print(Points_2D)
    M = np.linalg.lstsq(A,b)[0];
    M = np.append(M,1)
    M = np.reshape(M, (3,4))

    return M

# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates
def compute_camera_center(M):
    ##################
    # Your code here #
    ##################

    # Replace this with the correct code
    # In the visualization you will see that this camera location is clearly
    # incorrect, placing it in the center of the room where it would not see all
    # of the points.
    Q=M[:,0:3]
    inv=-(np.linalg.inv(Q))
    Center = np.dot(inv,M[:,3])

    return Center

# Returns the camera center matrix for a given projection matrix
# 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
# 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
# 'F_matrix' is 3x3 fundamental matrix
def estimate_fundamental_matrix(Points_a,Points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project
    ##################
    # Your code here #
    ##################

    # This is an intentionally incorrect Fundamental matrix placeholder
    sa=1.5*np.reciprocal(np.sqrt(np.sum((Points_a[:,0]-np.mean(Points_a[:,0]))**2+(Points_a[:,1]-np.mean(Points_a[:,1]))**2)/Points_a.shape[0]))
    sb=1.5*np.reciprocal(np.sqrt(np.sum((Points_b[:,0]-np.mean(Points_b[:,0]))**2+(Points_b[:,1]-np.mean(Points_b[:,1]))**2)/Points_b.shape[0]))
    ta=np.array([[sa,0,0],[0,sa,0],[0,0,1]])@np.array([[1,0,-np.mean(Points_a[:,0])],[0,1,-np.mean(Points_a[:,1])],[0,0,1]])
    tb=np.array([[sb,0,0],[0,sb,0],[0,0,1]])@np.array([[1,0,-np.mean(Points_b[:,0])],[0,1,-np.mean(Points_b[:,1])],[0,0,1]])
    Points_a=np.transpose(ta@np.transpose(np.hstack((Points_a,np.ones((Points_a.shape[0],1))))))
    Points_b=np.transpose(tb@np.transpose(np.hstack((Points_b,np.ones((Points_b.shape[0],1))))))
    Points_a=Points_a[:,0:2]
    Points_b=Points_b[:,0:2]
    F_matrix = np.array([[1,1,1],[1,1,2],[1,1,1]])
    A=np.ones((Points_a.shape[0],9))
    for x in range(Points_a.shape[0]):
        A[x]=[Points_a[x][0]*Points_b[x][0],Points_a[x][0]*Points_b[x][1],Points_a[x][0],
        Points_a[x][1]*Points_b[x][0],Points_a[x][1]*Points_b[x][1],Points_a[x][1],Points_b[x][0],Points_b[x][1],1]
    U, S, Vh = np.linalg.svd(A)
    #print(U,S,Vh)
    F = Vh[-1,:]
    F= np.transpose(np.reshape(F, (3,3)))
    U, S, Vh = np.linalg.svd(F)
    S[-1]=0
    F_matrix = U @ np.diagflat(S) @ Vh
    #print("f mat ")
    #print(F_matrix)

    return np.transpose(tb)@F_matrix@ta
    #return F_matrix

# Takes h, w to handle boundary conditions
def apply_positional_noise(points, h, w, interval=3, ratio=0.2):
    """ 
    The goal of this function to randomly perturbe the percentage of points given 
    by ratio. This can be done by using numpy functions. Essentially, the given 
    ratio of points should have some number from [-interval, interval] added to
    the point. Make sure to account for the points not going over the image 
    boundary by using np.clip and the (h,w) of the image. 
    
    Key functions include but are not limited to:
        - np.random.rand
        - np.clip

    Arugments:
        points :: numpy array 
            - shape: [num_points, 2] ( note that it is <x,y> )
            - desc: points for the image in an array
        h :: int 
            - desc: height of the image - for clipping the points between 0, h
        w :: int 
            - desc: width of the image - for clipping the points between 0, h
        interval :: int 
            - desc: this should be the range from which you decide how much to
            tweak each point. i.e if interval = 3, you should sample from [-3,3]
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will have some number from 
            [-interval, interval] added to the point. 
    """
    ##################
    # Your code here #
    ##################
    for x in range(int(points.shape[0]*ratio)):
        points[x][0]+=random.randrange(-interval, interval)
        points[x][1]+=random.randrange(-interval, interval)
        points[x][0]=np.clip(points[x][0], a_min =0, a_max = w)
        points[x][1]=np.clip(points[x][1], a_min =0, a_max = h)

    return points

# Apply noise to the matches. 
def apply_matching_noise(points, ratio=0.2):
    """ 
    The goal of this function to randomly shuffle the percentage of points given 
    by ratio. This can be done by using numpy functions. 
    
    Key functions include but are not limited to:
        - np.random.rand
        - np.random.shuffle  

    Arugments:
        points :: numpy array 
            - shape: [num_points, 2] 
            - desc: points for the image in an array
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will be randomly shuffled.
    """
    ##################
    # Your code here #
    ##################
    np.random.shuffle(points[0:int(points.shape[0]*ratio)])
    return points


# Find the best fundamental matrix using RANSAC on potentially matching
# points
# 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
# matching points from pic_a and pic_b. Each row is a correspondence (e.g.
# row 42 of matches_a is a point that corresponds to row 42 of matches_b.
# 'Best_Fmatrix' is the 3x3 fundamental matrix
# 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
# of 'matches_a' and 'matches_b') that are inliers with respect to
# Best_Fmatrix.
def ransac_fundamental_matrix(matches_a, matches_b):
    # For this section, use RANSAC to find the best fundamental matrix by
    # randomly sampling interest points. You would reuse
    # estimate_fundamental_matrix() from part 2 of this assignment.
    # If you are trying to produce an uncluttered visualization of epipolar
    # lines, you may want to return no more than 30 points for either left or
    # right images.
    ##################
    # Your code here #
    ##################

    # Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.

    # placeholders, you can delete all of this
    Best_Fmatrix = estimate_fundamental_matrix(matches_a[0:9,:],matches_b[0:9,:])
    inliers_a = matches_a[0:29,:]
    inliers_b = matches_b[0:29,:]
    combo_ab=np.ones((matches_a.shape[0],4))
    combo_ab[:,0]=matches_a[:,0]
    combo_ab[:,1]=matches_a[:,1]
    combo_ab[:,2]=matches_b[:,0]
    combo_ab[:,3]=matches_b[:,1]
    
    s_a=np.ones((matches_a.shape[0],2))
    s_b=np.ones((matches_a.shape[0],2))

    thresold=0.1
    max_inlier_counter=0
    for i in range(5000):
        sampling=random.randrange(8, 9)
        inlier_counter=0
        #shuffle (a,b)---> a,b 
        np.random.shuffle(combo_ab)
        s_a[:,0]=combo_ab[:,0]
        s_a[:,1]=combo_ab[:,1]
        s_b[:,0]=combo_ab[:,2]
        s_b[:,1]=combo_ab[:,3]
        tem_Fmatrix=estimate_fundamental_matrix(s_a[0:sampling,:],s_b[0:sampling,:])
        ina_tmp=np.ones((1,2))
        inb_tmp=np.ones((1,2))
        for m in range(matches_a.shape[0]):
            bi=np.transpose(np.append(matches_b[m],1))
            ai=np.append(matches_a[m],1)
            distance=bi@tem_Fmatrix@ai
            if abs(distance)<thresold:
                inlier_counter+=1
                ina_tmp= np.append(ina_tmp,matches_a[m])
                inb_tmp= np.append(inb_tmp,matches_b[m])
        ina_tmp=ina_tmp[2:ina_tmp.shape[0]].reshape((inlier_counter,2))
        inb_tmp=inb_tmp[2:inb_tmp.shape[0]].reshape((inlier_counter,2))        
        
        if inlier_counter>max_inlier_counter:
            max_inlier_counter=inlier_counter
            Best_Fmatrix=tem_Fmatrix
            inliers_a=ina_tmp
            inliers_b=inb_tmp


    #[0:20,:]


    return Best_Fmatrix, inliers_a, inliers_b
