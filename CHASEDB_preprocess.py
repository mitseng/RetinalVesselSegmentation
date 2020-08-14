# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:51:56 2020

@author: zll
"""


import os
import cv2 as cv
import numpy as np
from preprocessing import my_proc


read_path = '../data/CHASEDB/train/image/'
write_path = '../data/CHASEDB/train/proc_imgs/'


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
