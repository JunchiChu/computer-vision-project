# Project Image Filtering and Hybrid Images - Generate Hybrid Image
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from helpers import vis_hybrid_image, load_image, save_image

from student import my_imfilter, gen_hybrid_image

# Before trying to construct hybrid images, it is suggested that you
# implement my_imfilter in helpers.py and then debug it using proj1_part1.py

# Debugging tip: You can split your python code and print in between
# to check if the current states of variables are expected.

import numpy as np
from  skimage.transform import downscale_local_mean
from skimage import data
from skimage.transform import resize
# Read images and convert to floating point format
from PIL import Image

image1 = load_image('../junch/cv/nba kobe/e.bmp')
image2 = load_image('../junch/cv/nba kobe/m.bmp')
print(image1.shape)
image1=resize(image1,(1000,1000,3))
image2=resize(image2,(1000,1000,3))


# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# import matplotlib.pyplot as plt
# import numpy as np
# from  skimage.transform import downscale_local_mean
# from skimage import data
# from skimage.transform import resize
# import time
# from  scipy import ndimage
# import random

# from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D
# fig = pyplot.figure()
# ax = Axes3D(fig)
# timelist=[]
# imagesize=[]
# #ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
# filter_list=[3,5,7,9,11,13,15]
# image_o = load_image('../junch/cv/RISDance.jpg')
# image = image_o
# for m in filter_list:

#     for x in range(10):
       
#        k=np.ones((m,m,3))
#        imagesize.append(image.shape[0]*image.shape[1]*0.000001)
#        start=time.time()
#        ndimage.convolve(image, k, mode='constant', cval=0.0)
#        end=time.time()
#        timelist.append(end-start)
#        image=resize(image,(int(image.shape[0]-177.5),int(image.shape[1]-315.5),3))
#     ax.scatter([m,m,m,m,m,m,m,m,m,m], imagesize, timelist, c='r', marker='o')   
#     imagesize=[]
#     timelist=[]
#     image= image_o

       
# ax.set_xlabel('filter_size-Int ')
# ax.set_ylabel('image_size-Millon Pixels')
# ax.set_zlabel('Time-Seconds')
       
# pyplot.show() 
   



# display the dog and cat images

plt.figure(figsize=(3, 3))
plt.imshow((image1*255).astype(np.uint8))
plt.figure(figsize=(3, 3))
plt.imshow((image2*255).astype(np.uint8))

# For your write up, there are several additional test cases in 'data'.
# Feel free to make your own, too (you'll need to align the images in a
# photo editor such as Photoshop).
# The hybrid images will differ depending on which image you
# assign as image1 (which will provide the low frequencies) and which image
# you asign as image2 (which will provide the high frequencies)

## Hybrid Image Construction ##
# cutoff_frequency is the standard deviation, in pixels, of the Gaussian#
# blur that will remove high frequencies. You may tune this per image pair
# to achieve better results.
cutoff_frequency = 7
low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(
    image1, image2, cutoff_frequency)
## Visualize and save outputs ##


plt.figure()
plt.imshow((low_frequencies*255).astype(np.uint8))
plt.figure()
plt.imshow(((high_frequencies+0.5)*255).astype(np.uint8))
vis = vis_hybrid_image(hybrid_image)
plt.figure(figsize=(20, 20))
plt.imshow(vis)

save_image('../junch/cv/nba kobe/low_frequencies.jpg', low_frequencies)
outHigh = np.clip(high_frequencies + 0.5, 0.0, 1.0)
save_image('../junch/cv/nba kobe/high_frequencies.jpg', outHigh)
save_image('../junch/cv/nba kobe/hybrid_image.jpg', hybrid_image)
save_image('../junch/cv/nba kobe/hybrid_image_scales.jpg', vis)
