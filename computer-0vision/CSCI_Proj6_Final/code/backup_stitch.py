import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import io 
# from scipy import ndimage
# from pylab import *
# from PCV.geometry import warp


def stitch(imageA, imageB, ptsA, ptsB, reprojThresh):
    # compute the homography between the two sets of points
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
    # print('H\n',H)

    # stitching 2 images
    stitched_image = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    #plt.imshow(stitched_image),plt.show()
    covered = stitched_image[0:imageB.shape[0], 0:imageB.shape[1]]
    stitched_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    
    return stitched_image, covered
