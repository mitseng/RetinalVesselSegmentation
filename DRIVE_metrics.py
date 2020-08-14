# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:42:42 2020

@author: zll
"""

from metrics import metrics, roc
import os
from patch_extracting import get_imarr
import numpy as np

#######################################
predict_path = ''
lable_path = ''
mask_path = ''
out_path = ''
#######################################


# get file name from  paths
pred_files = os.listdir(predict_path)
lable_files = os.listdir(lable_path)
mask_files = os.listdir(mask_path)

# sort the file name list
pred_files.sort()
lable_files.sort()
mask_files.sort()

# combine image path with file names
pred_files = [predict_path + f for f in pred_files]
lable_files = [lable_path + f for f in lable_files]
mask_files = [mask_path + f for f in mask_files]

# get array of image array
pred_imgs = get_imarr(pred_files)
lable_imgs = get_imarr(lable_files)
# mask_imgs = get_imarr(mask_files)  # needless
outs = np.load(out_path)
print('data loaded.')

met = metrics(pred_imgs, lable_imgs)
auroc = roc(lable_imgs, outs)
print(met)
