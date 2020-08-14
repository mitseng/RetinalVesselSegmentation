# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:19:35 2020

@author: zll
"""

import cv2 as cv
import numpy as np
import imageio
import torch
from preprocessing import my_proc
from MF_UNet2 import MF_U_Net
from U_Net2 import U_Net
from math import floor, ceil
from metrics import metrics, roc


def get_patchs(img):
    '''
    extract patch from image for segmetation
    '''
    cols, rows = img.shape
    patchs = list()
    x, y = 0, 0  # position of left-up corner
    for i in range(cols // 52):
        for j in range(rows // 52):
            patch = img[x: x + 52, y: y + 52]
            patch = np.pad(patch, 6, "mean")
            patchs.append(patch)
            y += 52
        y = 0
        patch = img[x: x + 52, -52:]
        patch = np.pad(patch, 6, "mean")
        patchs.append(patch)
        x += 52
    x += 0
    for j in range(rows // 52):
        patch = img[-52:, y: y + 52]
        patch = np.pad(patch, 6, "mean")
        patchs.append(patch)
    patch = img[-52:, -52:]
    patch = np.pad(patch, 6, "mean")
    patchs.append(patch)

    patchs = np.array(patchs)
    return patchs


def from_patch(patchs, img_shape):
    '''
    input:patchs:numpy array
    '''
    index = 0  # patch index
    cols, rows = img_shape
    img = np.empty(img_shape, dtype='uint8')
    x, y = 0, 0  # position of left-up corner
    for i in range(cols // 52):
        for j in range(rows // 52):
            img[x+6:x+58, y+6:y+58] = patchs[index, 6:-6, 6:-6]
            index += 1
            y += 52
        y = 0
        img[x+6:x+58, -58:] = patchs[index, 6:-6, 6:]
        index += 1
        x += 52
    for j in range(rows // 52):
        img[-58:, y+6:y+58] = patchs[index, 6:, 6:-6]
        index += 1
    img[-58:, -58:] = patchs[index, 6:, 6:]
    index += 1

    return img


def seg_img_patch(net, img_file, mask_file=None):
    # read images
    img = cv.imread(img_file)
    if img is None:    # cv2 cannot read gif
        img = imageio.mimread(img_file)  # but imageio can
    if mask_file is not None:
        msk = cv.imread(mask_file, cv.IMREAD_GRAYSCALE)
        if msk is None:
            msk = imageio.mimread(mask_file)[0]

    # preprocess
    proc = my_proc(np.uint8([img]))[0]  # numpy array of uint8

    inputs = get_patchs(proc)

    inputs = inputs.reshape(inputs.shape[0], 1,
                            inputs.shape[1], inputs.shape[2])

    # normalization
    inputs = np.float32(inputs)
    inputs /= 255

    inputs = torch.tensor(inputs)

    out = net(inputs)[:, 1, :, :]

    out = np.uint8(out >= 0.5)

    out = from_patch(out, proc.shape)

    pred = out

    pred = np.uint8(pred >= 0.5)
    pred *= 255

    # mask
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred[i, j] = pred[i, j] if msk[i, j] == 255 else 0

    return pred


def seg_img_patch4(net, img_file, mask_file=None):
    # read images
    img = cv.imread(img_file)
    if img is None:    # cv2 cannot read gif
        img = imageio.mimread(img_file)  # but imageio can
    if mask_file is not None:
        msk = cv.imread(mask_file, cv.IMREAD_GRAYSCALE)
        if msk is None:
            msk = imageio.mimread(mask_file)[0]

    # preprocess
    proc = my_proc(np.uint8([img]))[0]  # numpy array of uint8
    pred = np.zeros(proc.shape, dtype='uint8')  # predict

    for rot in range(4):  # rotate 4 times
        rot_pic = np.rot90(proc, rot)

        inputs = get_patchs(rot_pic)

        inputs = inputs.reshape(inputs.shape[0], 1,
                                inputs.shape[1], inputs.shape[2])

        # normalization
        inputs = np.float32(inputs)
        inputs /= 255

        inputs = torch.tensor(inputs)

        out = net(inputs)[:, 1, :, :]

        out = np.uint8(out >= 0.5)

        out = from_patch(out, rot_pic.shape)

        # rotate back
        out = np.rot90(out, 4 - rot)
        pred += out

    pred = np.uint8(pred >= 2)
    pred *= 255

    # mask
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred[i, j] = pred[i, j] if msk[i, j] == 255 else 0

    return pred


def seg_img(net, img_file, mask_file=None):
    '''
    arguments:
        net: initialized model
        img_file: str, image(before preprocess) file name
        mask_file: str, mask(if there is) file name
    returns:
        (pred: segment prediction, 2-D numpy byte array, 0 and 255, \
         out: net output before thresholding, 2-D numpy float32 array)
    '''
    # read images
    img = cv.imread(img_file)
    if img is None:    # cv2 cannot read gif
        img = imageio.mimread(img_file)  # but imageio can
    if mask_file is not None:
        msk = cv.imread(mask_file, cv.IMREAD_GRAYSCALE)
        if msk is None:
            msk = imageio.mimread(mask_file)[0]

    # preprocess
    proc = my_proc(np.uint8([img]))[0]  # numpy array of uint8
    shape = ((proc.shape[0] // 8 + 1) * 8,
             (proc.shape[1] // 8 + 1) * 8)  # shape of inputs
    inputs = np.empty(shape, dtype='float32')  # inputs of net
    # may need padding
    pad = (floor((shape[0] - proc.shape[0]) / 2),
           ceil((shape[0] - proc.shape[0]) / 2),
           floor((shape[1] - proc.shape[1]) / 2),
           ceil((shape[1] - proc.shape[1]) / 2))
    inputs[pad[0]: -pad[1], pad[2]: -pad[3]] = proc
    # convert to 4-D
    inputs = inputs.reshape((1, 1, shape[0], shape[1]))
    # normalization
    inputs /= 255
    inputs = torch.tensor(inputs)

    out = net(inputs)
    out = out[0, 1, :, :]
    # remove padding
    out = out[pad[0]: -pad[1], pad[2]: -pad[3]]
    # prediction, thresholding by 0.5
    pred = np.uint8(out >= 0.5)
    pred *= 255
    out = out.detach().numpy()

    # mask
    if mask_file is not None:
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred[i, j] = pred[i, j] if msk[i, j] == 255 else 0

    return pred, out  # out may be used in roc


if __name__ == '__main__':
    model = MF_U_Net()
    model.eval()
    model.load_state_dict(torch.load('mf_unet2_400.pkl',
                                     map_location=torch.device('cpu')))
    
    pred, out = seg_img(model,
                        '../data/DRIVE/test/proc_imgs/01_test.tif',
                        '../data/DRIVE/test/mask/01_test_mask.gif')
    '''
    pred, out = seg_img(model,
                        '../data/Image_01L.jpg')
    '''
    cv.imshow('img', pred)
    cv.waitKey(0)
    cv.destroyAllWindows()
    lable_file = '../data/DRIVE/test/1st_manual/01_manual1.gif'
    #lable_file = '../data/Image_01L_1stHO.png'
    target = cv.imread(lable_file, cv.IMREAD_GRAYSCALE)
    if target is None:    # cv2 cannot read gif
        target = imageio.mimread(lable_file)[0]  # but imageio can
    met = metrics(pred, target)
    auroc = roc(target, out)
