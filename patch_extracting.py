# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:15:27 2020

@author: zll
"""


import cv2 as cv
import imageio
import numpy as np


def get_imarr(files):
    '''
    return image array consist of input files.
    input: list of string, file name.
    return: numpy.ndarray.
    '''
    imgs = list()
    for f in files:
        # read images
        img = cv.imread(f, cv.IMREAD_GRAYSCALE)
        if img is None:    # cv2 cannot read gif
            img = imageio.mimread(f)[0]  # but imageio can
        imgs.append(img)   # add to image list
    imgs = np.uint8(imgs)  # convert to numpy array
    return imgs


def extract(img_arr, patch_size, step, is_lable):
    '''
    extract patch from images.
    rotate patch for augmentation(4 times).
    input: img_arr: 3D-numpy array
           patch_size: tuple
           step: integer
    returns numpy 3D-array
    '''
    # patch size: height, width
    h, w = patch_size
    # rows of patch in an image
    rows = (img_arr.shape[1] - h) // step
    # columns of patch in an image
    cols = (img_arr.shape[2] - w) // step
    # number of patchs
    N = img_arr.shape[0] * rows * cols * 4
    if is_lable:
        patch_array = np.zeros(tuple([N]) + patch_size, dtype='int32')
    else:
        patch_array = np.zeros(tuple([N]) + patch_size, dtype='float32')
    patch_index = 0  # index in patch_array
    for img in img_arr:  # walk image list
        # normalize
        img = np.float32(img)
        img = img / 255
        r, c = 0, 0      # up-left coordination of patch
        for i in range(rows):
            c = 0
            for j in range(cols):
                patch_array[patch_index] = img[r:r+w, c:c+h]
                patch_index += 1
                for k in range(3):  # rotate 3 times
                    # rotate 90 degrees
                    patch_array[patch_index] =\
                        np.rot90(patch_array[patch_index-1], 1)
                    patch_index += 1
                c += step  # move horizontally
            r += step      # move vertically
    return patch_array
