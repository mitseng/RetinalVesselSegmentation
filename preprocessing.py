# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:52:52 2020

@author: zll
"""

import cv2 as cv
import numpy as np


def to_gray(img):
    '''
    convert RGB picture to gray
    using cv2
    '''
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def my_to_gray(img):
    '''convert to gray using my algorithm'''
    max = np.linalg.norm([255, 255, 255])
    gray_img = np.zeros(img.shape[:2], dtype=img.dtype)
    for i in range(img.shape[0]):  # iterate by row
        for j in range(img.shape[1]):  # iterate by colum
            gray_img[i][j] = round(np.linalg.norm(img[i][j]) / max * 255)
    return gray_img


def all_to_gray(imgs):
    '''
    convert a list fo imgs to gray
    input: np.ndarray
    '''
    gray_imgs = list()
    for i in range(imgs.shape[0]):
        # using cv
        gray_imgs.append(to_gray(imgs[i]))
    gray_imgs = np.uint8(gray_imgs)
    return gray_imgs

def standardize(input_imgs):
    '''
    Z-score standardization.
    input: numpy.ndarray
    '''
    imgs = np.float64(input_imgs)
    mean = np.mean(imgs)  # get mean of all images
    std = np.std(imgs)    # get standard deviation of all images
    imgs_standardized = (imgs - mean) / std  # standardize
    for i in range(imgs_standardized.shape[0]):
        # scale to [0, 1]
        imgs_standardized[i] = (imgs_standardized[i] - np.min(imgs_standardized[i])) / (np.max(imgs_standardized[i]) - np.min(imgs_standardized[i]))
    return imgs_standardized


def clahe(imgs):
    '''
    contrast limited adaptive histogram equelization.
    using clahe in OpenCV2
    '''
    algo = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # clahe process
    imgs_clahe = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_clahe[i] = algo.apply(imgs[i])
    return imgs_clahe


def gamma_adjust(imgs, gamma=1.0):
    '''
    gamma adjustment.
    set gamma > 1, to enhance the contrast in low intensity areas,
    where is vessels.
    '''
    gamma_inv = 1.0 / gamma  # inverse of gamma
    # creat a look up table.
    table = [(i / 255) ** gamma_inv * 255 for i in range(0, 256)]
    table = np.uint8(table)
    imgs_adjusted = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        imgs_adjusted[i] = cv.LUT(np.uint8(imgs[i]), table)
    return imgs_adjusted


def my_proc(imgs):
    '''
    my preprocess function.
    input: np.ndarray
    '''
    imgs = all_to_gray(imgs)      # convert to gray image
    imgs = standardize(imgs)      # Z-standardize
    imgs *= 255                   # scale to [0, 255]
    imgs = np.uint8(imgs)
    imgs = np.uint8(clahe(imgs))  # clahe
    imgs = np.uint8(gamma_adjust(imgs, 1.2))  # gamma adjustment
    return imgs


'''
for f in read_files:
    img = cv2.imread(read_path+f)  # read image
    img = to_gray(img)  # convert to gray
    cv2.imwrite(write_path+f, img)
'''
'''
imgs = list()
for f in read_files:
    img = cv2.imread(read_path+f)  # read image
    imgs.append(img)
imgs_standardized = standardize(np.array(imgs))
for i, f in enumerate(read_files):
    cv2.imwrite(write_path+f, imgs_standardized[i])
'''
'''
imgs = list()
for f in read_files:
    img = cv2.imread(read_path+f)
    img = to_gray(img)
    imgs.append(img)
imgs = np.uint8(imgs)
imgs_clahe = clahe(imgs)
imgs_clahe = np.uint8(imgs_clahe)
for i, f in enumerate(read_files):
    cv2.imwrite(write_path+f, imgs_clahe[i])
'''
'''
imgs = list()  # image list
for f in read_files:
    img = cv2.imread(read_path+f)  # read file
    img = to_gray(img)  # convert to gray image
    imgs.append(img)
imgs = np.uint8(imgs)
imgs_adjusted = gamma_adjust(imgs, gamma=1.2)
imgs_adjusted = np.uint8(imgs_adjusted)
for i, f in enumerate(read_files):
    cv2.imwrite(write_path+f, imgs_adjusted[i])
'''
'''
cv2.imshow('img', imgs_adjusted[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
