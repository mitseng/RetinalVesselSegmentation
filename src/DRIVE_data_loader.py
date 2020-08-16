# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:55:25 2020

@author: zll
this file tests if data set work
"""

from DRIVE_data_set import DRIVE_data_set
from torch.utils.data import DataLoader
import numpy as np

dds = DRIVE_data_set(train=True)

loader = DataLoader(dds,
                    batch_size=8)

imgs = list()
lables = list()
for i, data in enumerate(loader, 0):
    img, lable = data
    imgs.append(np.uint8(img))
    lables.append(np.uint8(lable))
    if i > 1:
        break
