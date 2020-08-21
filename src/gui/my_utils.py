import cv2 as cv
from PIL import Image
import numpy as np
from math import floor, ceil
import torch
from Util_Interface import Util_Interface
import sys
sys.path.append('..')
from preprocessing import my_proc
from MF_UNet2 import MF_U_Net


class Util(Util_Interface):
    def __init__(self):
        # load model
        self. model = MF_U_Net()
        self.model.eval()
        self.model.load_state_dict(torch.load('../mf_unet2_180.pkl',
                                        map_location=torch.device('cpu')))

    
    def predict(self, img_file):
        '''
        arguments:
            img_file: str, image(before preprocess) file name
        returns:
            (pred: segment prediction, 2-D numpy byte array, 0 and 255, \
            out: net output before thresholding, 2-D numpy float32 array)
        '''
        # read images
        img = cv.imread(img_file)
        if img is None:    # cv2 cannot read gif
            img = Image.open(img_file)  # but imageio can
            img = np.array(img)

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

        out = self.model(inputs)
        out = out[0, 1, :, :]
        # remove padding
        out = out[pad[0]: -pad[1], pad[2]: -pad[3]]
        # prediction, thresholding by 0.5
        pred = np.uint8(out >= 0.5)
        pred *= 255

        return pred