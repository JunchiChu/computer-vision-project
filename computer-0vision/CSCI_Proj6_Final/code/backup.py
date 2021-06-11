import cv2
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import io 
from mystitch import *

def FindMatchedPoints(img1, img2, extract_func, num_features, ToPlot = False):

    # choose from three feature extraction methods
    if extract_func is "SIFT" or "SURF":
        src_xy_coord, dst_xy_coord = FeatureWithSIFTorSURF(img1, img2, num_features, extract_func, ToPlot)

    elif extract_func is "ORB":
        src_xy_coord, dst_xy_coord = FeatureWithORB(img1, img2, num_features, ToPlot)

    else:
        sys.exit("Please choose a valid func: SIFT, SURF, ORB")

    return src_xy_coord, dst_xy_coord


# Extract Feature Pairs with SIFT/SURF + FLANN + RATIO TEST
def FeatureWithSIFTorSURF(img1, img2, num_features, extract_func, ToPlot):
    #print(type(extract_func),extract_func)
    # extract features with SIFT
    if extract_func == "SIFT":
        num_features+=2000
        sift = cv2.xfeatures2d.SIFT_create(num_features)
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
      


    # extract features with SURF
    else:
        surf = cv2.xfeatures2d.SURF_create(num_features)
        kp1, des1 = surf.detectAndCompute(img1, None)
        kp2, des2 = surf.detectAndCompute(img2, None)
        


    # source code from https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)

    # convert the matched key points to xy coordinates
    src_xy_coord = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
    dst_xy_coord = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)

    # optionally plot the two images
    if(ToPlot):
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None)
        #plt.imshow(img3, 'gray'),plt.show()

    return src_xy_coord, dst_xy_coord


# Extract Feature Pairs with ORB + Brute Force
def FeatureWithORB(img1, img2, num_features, ToPlot):

    # extract features
    orb = cv2.ORB_create(num_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # convert the matched key points to xy coordinates
    src_xy_coord = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,2)
    dst_xy_coord = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,2)

    # optionally plot the two images
    if(ToPlot):
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None)
        #plt.imshow(img3, 'gray'),plt.show()

    return src_xy_coord, dst_xy_coord

def refine_image(img, width):
    (_, mask) = cv2.threshold(img, 1.0, 255.0, cv2.THRESH_BINARY_INV);
    temp = mask.copy()
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    (_, contours, _) = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contour:len(contour), reverse=True)
    roi = cv2.boundingRect(contours[0])
    if roi[0] > width:
        result = img[:, :roi[0], :]
    else:
        result = img
    plt.imshow(result), plt.show()
    return result

# sort contours by largest first (if there are more than one)
    contours = sorted(contours, key=lambda contour:len(contour), reverse=True)
    roi = cv2.boundingRect(contours[0])
    result = img1[:, roi[0]:roi[2], :]
    plt.imshow(result), plt.show()
    return result

if __name__ == '__main__':

    ######################################
    # load test image panorama-data1
    ######################################
    #img1_dir = '../data/panorama-data1/d (3).JPG'
    #img2_dir = '../data/panorama-data1/d (4).JPG'
    #img3_dir = '../data/panorama-data1/d (5).JPG'
    #img4_dir = '../data/panorama-data1/d (6).JPG'

    #img1 = io.imread(img1_dir)
    #img2 = io.imread(img2_dir)
    #img3 = io.imread(img3_dir)
    #img4 = io.imread(img4_dir)

    # define the feature extraction method here
    extract_func = input("Enter a extrac function! Your choice: SIFT,SURF,ORB: ")
    while extract_func not in ['SIFT','ORB','SURF']:
        extract_func = input("Enter a VALID extrac function! Your choice: SIFT,SURF,ORB:")
    print("extract function is",extract_func)

    # define how many feature points we want to extract
    num_features = 20

    # get the xy coordinated of the matched pairs

    # src_xy_coord, dst_xy_coord = FindMatchedPoints(img2, img4, extract_func, num_features, ToPlot = True)
    # result,covered = stitch_left(img4, img2, dst_xy_coord, src_xy_coord, reprojThresh = 3.0)
    # # result = stitch2(img2, img4, dst_xy_coord, src_xy_coord, reprojThresh = 3.0)
    # print(result)
    # plt.imshow(result),plt.show()
    # print(fffff)

    # src_xy_coord, dst_xy_coord = FindMatchedPoints( img2, result, extract_func, num_features, ToPlot = True)
    # result,covered = stitch( result, img2, dst_xy_coord, src_xy_coord, reprojThresh = 3.0)
    # plt.imshow(result),plt.show()

    # src_xy_coord, dst_xy_coord = FindMatchedPoints( img1, result, extract_func, num_features, ToPlot = True)
    # result,covered = stitch(result, img1, dst_xy_coord, src_xy_coord, reprojThresh = 3.0)
    # plt.imshow(result),plt.show()

    ######################################
    # load another set of image
    ######################################
    
    # img_dir = ['../data/road view/1.JPG']
    # img_dir.append('../data/road view/2.JPG')
    # img_dir.append('../data/road view/3.JPG')
    # img_dir.append('../data/road view/4.JPG')
    # image_cnt = len(img_dir) 

    # # for hoirzontal panarama, the image set goes from right to left
    # for i in range(image_cnt-1):
    #     img = io.imread(img_dir[image_cnt - i - 2])
    #     if i == 0:
    #         result = io.imread(img_dir[image_cnt - i - 1])
    #     src_xy_coord, dst_xy_coord = FindMatchedPoints(img, result,  extract_func, num_features, ToPlot = True)
    #     result,covered = stitch(result, img, dst_xy_coord, src_xy_coord, reprojThresh = 3.0)
    #     plt.imshow(result),plt.show()


    ######################################
    # load another set of image, using laoding code from George
    ######################################
    photo_cat=['bridge','car','flowers','greens','house1','house2','HouseAndRoad','RoadView','sofa','TV','house1','trial_data1','panorama-data1','panorama-data2','kp_image','waterman']
    parainput=input("Pick one: bridge/ car/ flowers/ greens/ house1/ house2/ HouseAndRoad/ RoadView/ sofa / TV / house1 / trial_data1/ panorama-data1/ panorama-data2/ kp_image/ waterman: ")
    while parainput not in photo_cat:
        parainput=input("Pick A VALID one: bridge/ car/ flowers/ greens/ house1/ house2/ HouseAndRoad/ RoadView/ sofa / TV / house1 / trial_data1/ panorama-data1/ panorama-data2/ kp_image/ waterman:: ")
    print("You select [",parainput,"] as input")

    # initialize string list
    img_dir = []

    # read all the image in the folder
    directory = r"../data/"+parainput
    lst = os.listdir(directory)
    lst.sort()
    for filename in lst:
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
            #print("sssss")
            print(os.path.join(directory, filename))
            img_dir.append(os.path.join(directory, filename))
        else:
            continue
    
    # print out the number of images
    image_cnt = len(img_dir) 
    print("No. of images read in:",image_cnt)

    for i in range(image_cnt):
        if i == 0:
            result = io.imread(img_dir[i])
            continue
        img = io.imread(img_dir[i])
        width = result.shape[1]
        src_xy_coord, dst_xy_coord = FindMatchedPoints(result, img,  extract_func, num_features, ToPlot = True)
        result = stitch(img, result, dst_xy_coord, src_xy_coord, reprojThresh = 1.0)
        plt.imshow(result), plt.show()
        result=refine_image(result, width)
        #plt.imshow(result),plt.show()
    print(result.shape,"result")
    #result=refine_image(result)
    plt.imshow(result)
    #plt.show()
    plt.savefig(directory+"/myresult.PNG")
    
 

