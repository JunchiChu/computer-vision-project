import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import io 
from PIL import Image
# from scipy import ndimage
# from pylab import *
# from PCV.geometry import warp

'''
def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)
    '''

# https://note.nkmk.me/en/python-numpy-generate-gradation-image/
def get_gradation_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T

def get_gradation_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradation_2d(start, stop, width, height, is_horizontal)

    return result

def stitch(imageA, imageB, ptsA, ptsB, reprojThresh):
    N = 100
    # compute the homography between the two sets of points
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
    # print('H\n',H)

    # stitching 2 images
    stitched_image = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    #print("A:", imageA.shape[1], "B:", imageB.shape[1], "S:", stitched_image.shape[1])
    blendA = stitched_image[:, imageB.shape[1]-N:imageB.shape[1],:]
    blendB = imageB[:, imageB.shape[1]-N:imageB.shape[1],:]
    '''
    print(blendA.shape, blendB.shape, stitched_image.shape)
    plt.axis('off'),plt.imshow(blendA), plt.savefig("tmp1.jpg", bbox_inches='tight', pad_inches=0)
    image1 = Image.open("tmp1.jpg")
    plt.axis('off'),plt.imshow(blendB), plt.savefig("tmp2.jpg", bbox_inches='tight', pad_inches=0)
    image2 = Image.open("tmp2.jpg")
    blend_image = Image.blend(image2, image1, 0.5)
    plt.imshow(blend_image),plt.savefig("tmp3.jpg")
    blended = io.imread("tmp3.jpg")
    print(blended.shape)
    '''
    maskarr = get_gradation_3d(N, imageA.shape[0], (0, 0, 0), (255, 255, 255), (True, True, True))
    #print(blendA.shape, blendB.shape, maskarr.shape)
    image1 = Image.fromarray(blendA, 'RGB')
    image2 = Image.fromarray(blendB, 'RGB')
    mask = Image.fromarray(maskarr, 'RGB')
    image1.putalpha(255)
    image2.putalpha(255)
    mask.putalpha(255)
    #print(image1.size, image2.size, mask.size)
    blend_image = Image.composite(image2, image1, mask)
    #blend_image.show()
    #print(blend_image.size)
    covered = stitched_image[0:imageB.shape[0], 0:imageB.shape[1]]
    stitched_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    stitched_image[:, imageB.shape[1]-N:imageB.shape[1]] = np.array(blend_image)[:,:,:3]
    
    return stitched_image, covered
