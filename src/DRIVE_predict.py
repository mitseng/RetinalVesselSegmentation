# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:30:36 2020

@author: zll
"""

import torch
import os
import numpy as np
import cv2 as cv
from U_Net2 import U_Net
from segmentation import seg_img

##########################################
# net path
net_path = 'unet2_150.pkl'
# preprocessed image path
img_path = ''
# mask image path
mask_path = ''
# save output of net for computing roc
out_path = ''
# path to save prediction image
pred_path = ''
##########################################


unet = U_Net()
unet.eval()
unet.load_state_dict(torch.load(net_path,
                                map_location=torch.device('cpu')))
print('net loaded.')


# get file name from  paths
img_files = os.listdir(img_path)
mask_files = os.listdir(mask_path)

# sort the file name list
img_files.sort()
mask_files.sort()

# combine image path with file names
img_files = [img_path + f for f in img_files]
mask_files = [mask_path + f for f in mask_files]

outs = list()
for i in range(len(img_files)):
    pred, out = seg_img(unet, img_files[i], mask_files[i])
    cv.imwrite(img_files[i], pred)  # save prediction
    outs.append(out)

outs = np.array(outs, dtype='float32')
np.save(out_path, outs)  # save outs




















