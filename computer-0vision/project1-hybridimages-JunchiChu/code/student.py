# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale

def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use numpy multiplication and summation
    when applying the kernel.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, k)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    
    if(np.mod(kernel.shape[0],2)==0 | np.mod(kernel.shape[1],2)==0):
       raise Exception('bad! filter/kernel has any even dimension')
    if(len(image.shape)==3):
        #print(image.shape)
        #print(image[:,:,0].shape)
        a1=my_imfilter(image[:,:,0],kernel)
        b1=my_imfilter(image[:,:,1],kernel)
        c1=my_imfilter(image[:,:,2],kernel)
        filter_image = np.zeros((image.shape[0],image.shape[1],image.shape[2]))
        filter_image[:,:,0]=a1
        filter_image[:,:,1]=b1
        filter_image[:,:,2]=c1
        return filter_image
        #fil=np.array([c1,b1,a1])
    else:
        kernel=np.rot90(kernel)
        kernel=np.rot90(kernel)
        pad_row=int((kernel.shape[0]-1)/2)
        pad_col=int((kernel.shape[1]-1)/2)
        #print(((pad_row, 0), (pad_col, 0)))
        image_pad = np.pad(image, ((pad_row, pad_row), (pad_col, pad_col)), mode='constant', constant_values=0) 
        #print(image_pad)
        filtered_image = np.zeros(image.shape)
        for i in range(pad_row,pad_row+image.shape[0]):  
           for j in range(pad_col,pad_col+image.shape[1]):
              # print(image_pad[i][j])
              #print(image_pad[i-pad_row:i+pad_row+1,j-pad_col:j+pad_col+1])
              #print(np.sum(np.multiply(image_pad[i-pad_row:i+pad_row+1,j-pad_col:j+pad_col+1],kernel)))
              filtered_image[i-pad_row][j-pad_col]=np.sum(np.multiply(image_pad[i-pad_row:i+pad_row+1,j-pad_col:j+pad_col+1],kernel))
              #print(filtered_image) 
        return filtered_image      

  

    ##################
    # Your code here #
    print('my_imfilter function in student.py needs to be implemented')
    ##################

    return filtered_image

"""
EXTRA CREDIT placeholder function
"""

def my_imfilter_fft(image, kernel):
    """
    Your function should meet the requirements laid out in the extra credit section on
    the project webpage. Apply a filter (using kernel) to an image. Return the filtered image.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, k)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
   
        
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    print('my_imfilter_fft function in student.py is not implemented')
    ##################

    return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    # Your code here:
    low_frequencies = np.zeros(image1.shape)
    low_frequencies = my_imfilter(image1,kernel) # Replace with your implementation

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    high_frequencies = np.zeros(image1.shape)
    blur_im2 = my_imfilter(image2,kernel)
    image2 = np.clip(image2, 0.0, 1.0)
    high_frequencies = image2 - blur_im2 # Replace with your implementation

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = np.zeros(image1.shape)
    hybrid_image = low_frequencies+high_frequencies # Replace with your implementation

    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to 
    # gen_hybrid_image().
    # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
    # and all values larger than 1.0 to 1.0.
    hybrid_image = np.clip(hybrid_image, 0.0, 1.0)
    return low_frequencies, high_frequencies, hybrid_image
