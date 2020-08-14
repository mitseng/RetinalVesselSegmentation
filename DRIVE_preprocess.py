# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:14:44 2020

@author: zll
"""


import os
import cv2 as cv
import numpy as np
from preprocessing import my_proc


mode = 'test'
read_path = '..\\data\\DRIVE\\' + mode + '\\images\\'
write_path = '..\\data\\DRIVE\\' + mode + '\\proc_imgs\\'


# get file name from read path
files = os.listdir(read_path)

imgs = list()
for f in files:
    # read images
    img = cv.imread(read_path+f)
    imgs.append(img)   # add to image list
imgs = np.uint8(imgs)  # convert to numpy array

imgs_proc = my_proc(imgs)  # process

# save images
for i, f in enumerate(files):
    cv.imwrite(write_path+f, imgs_proc[i])
