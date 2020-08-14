# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:36:12 2020

@author: zll
"""


import os
from patch_extracting import get_imarr, extract
import numpy as np

mode = 'test'

img_path = '..\\data\\DRIVE\\' + mode + '\\proc_imgs\\'
truth_path = '..\\data\\DRIVE\\' + mode + '\\1st_manual\\'
write_path = '..\\data\\DRIVE\\' + mode + '\\'


# get file name from image path
img_files = os.listdir(img_path)
# get file name from truth path
truth_files = os.listdir(truth_path)

# sort the file name list
img_files.sort()
truth_files.sort()

# combine image path with file names
img_files = [img_path + f for f in img_files]
truth_files = [truth_path + f for f in truth_files]


# get array
img_array = get_imarr(img_files)
truth_array = get_imarr(truth_files)

img_patchs = extract(img_array, (64, 64), 10, False)  # roughly 10k patchs per image
truth_patchs = extract(truth_array, (64, 64), 10, True)

np.save(write_path+'imgs', img_patchs)
np.save(write_path+'truth', truth_patchs)
