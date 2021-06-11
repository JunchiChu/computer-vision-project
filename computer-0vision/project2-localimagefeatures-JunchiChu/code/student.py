import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from scipy.ndimage import gaussian_filter
import scipy
from scipy import ndimage
from skimage.feature import peak_local_max
import math 
import copy

def get_interest_points(image, feature_width):
    '''
    Returns a set of interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - 
         (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!
    dx = ndimage.sobel(image,1)
    dy = ndimage.sobel(image,0)

    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    ky=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    dxx=ndimage.convolve(image, kx, mode='constant', cval=0.0)*0.25
    dyy=ndimage.convolve(image, ky, mode='constant', cval=0.0)*0.25
    # grad_x = filters.sobel_v(image)
    # grad_y = filters.sobel_h(image)

    C=np.multiply(gaussian_filter(dx**2, sigma=1),gaussian_filter(dy**2, sigma=1))-gaussian_filter(np.multiply(dx**2,dy**2), sigma=1)**2-0.05*(gaussian_filter(dx**2, sigma=1)+gaussian_filter(dy**2, sigma=1))**2
    C[C<0]=0
    C[abs(C)<0.00001]=0
 
    C=peak_local_max(C, min_distance=12,threshold_rel=0.0000001)
    #C=peak_local_max(C, min_distance=12,threshold_rel=0.0000001)

    value=C.shape[0]
    m=0
    
    
    while(m!=50):
        if C[m][0]-feature_width/2 <=0 or C[m][1]-feature_width/2 <=0:
            C=np.delete(C,m,0)
            value=value-1
        if C[m][0]+feature_width/2 >=image.shape[1] or C[m][1]+feature_width/2 >=image.shape[0]:
            C=np.delete(C,m,0)
            value=value-1
        m=m+1
    
    return C[:,1] , C[:,0]



def get_features(image, x, y, feature_width):
    '''
    Returns a set of feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    #print(x.shape)
    
    dx = filters.sobel_h(image)
    dy = filters.sobel_v(image)
    nori=np.arctan2(dy, dx) * 180 / np.pi+180
    #print(nori)
    features = np.zeros((1,128))
    #print(features)
    mag = np.sqrt(dx**2+dy**2)
    # for xx in range(mag.shape[0]):
    #     for yy in range(mag.shape[1]):
    #         mag[xx][yy]=np.linalg.norm(mag[xx][yy])
    fea=np.array([])
    for i in range(len(x)):
        #print(x[i])
        for a in range(int(y[i])+int(feature_width/2),int(y[i])-int(feature_width/2),-int(feature_width/4)):  
            for b in range(int(x[i])+int(feature_width/2),int(x[i])-int(feature_width/2),-int(feature_width/4)):
                b1=b2=b3=b4=b5=b6=b7=b8=0
                
                for m in range(a,a-int(feature_width/4),-1):
                    for n in range(b,b+int(feature_width/4),):
                
                        if 0 <= nori[m][n] <45:
                            b1=b1+mag[m][n]
                        elif 45<= nori[m][n] <90:
                            b2 = b2+mag[m][n]
                        elif 90<= nori[m][n] <135:
                            b3=b3+mag[m][n]
                        elif 135<= nori[m][n] <180:
                            b4=b4+mag[m][n]
                        elif 180<= nori[m][n] <225:
                            b5=b5+mag[m][n]
                        elif 225<= nori[m][n] <270:
                            b6=b6+mag[m][n]
                        elif 270<= nori[m][n] <315:
                            b7=b7+mag[m][n]
                        elif 315<= nori[m][n] <=360:
                            b8=b8+mag[m][n]
                fea=np.append(fea,[b1,b2,b3,b4,b5,b6,b7,b8])
        #print(features.shape)
        features=np.concatenate((features,[fea]))   
        fea=np.array([])  
    features=np.delete(features,0,0)
    # return np.linalg.norm(features)
    for aa in range(features.shape[0]):
        for bb in range(features.shape[1]):
            features[aa][bb]=np.linalg.norm(features[aa][bb])
    #print(features)
    #print(features)
    return features


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with your matches and confidences!
    matches = np.array ([[10,10]])
    confidences = np.array([1])
    # print(im1_features.shape)
    # print(im2_features.shape)
    B=2*np.dot(im1_features,im2_features.transpose())
    A=np.sum(im1_features**2,axis=1)[:,np.newaxis]+np.sum(im2_features**2,axis=1).transpose()[np.newaxis, :]
    # subA=np.sum(im1_features**2,axis=1)
    # A=subA[:, np.newaxis]+np.array([np.transpose(np.sum((im2_features)**2,axis=1))])
    D= np.sqrt(A-B)
    # print(D.shape)
    CD=copy.deepcopy(D)
    #print(D)
    for m in range(D.shape[0]):
        after_sort=np.argsort(CD[m])
        nn1 = nn2=0
        nn1 = D[m][after_sort[0]]
        nn2= D[m][after_sort[1]]
        if float(nn1/nn2) <0.95:
          matches=np.concatenate((matches,np.array([[m,after_sort[0]]])))
          confi_ratio = (float(nn2/nn1))
        #print(confi_ratio)
          confidences=np.concatenate((confidences,np.array([100*confi_ratio])))
    #confidences = np.ones([matches.shape[0]-1])
    #confidences = confidences.reshape((confidences.shape[0],1))
    matches=np.delete(matches,0,0)
    confidences=np.delete(confidences,0,0)
    # print("shape of match")
    #print(matches.shape)
    #print(matches)
    #print(confidences)
    #print(np.reshape(confidences,(matches.shape[0],1)).shape)
    # print(confidences.shape)
    # confidences=np.zeros((397,))
    # confidences=confidences+12
    #print(matches.shape)
    return matches, confidences
