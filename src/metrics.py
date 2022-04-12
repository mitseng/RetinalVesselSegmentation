# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:09:51 2020

@author: zll
"""

import PIL as Image
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def to_numpy(item):
    '''
    convert input to numpy array
    input type: image-file-name, PIL image, torch tensor, numpy array
    '''
    if 'str' in str(type(item)):  # read the image as numpy array
        item = np.array(Image.open(item))
    elif 'PIL' in str(type(item)):
        item = np.array(item)
    elif 'torch' in str(type(item)):
        item = item.numpy()
    elif 'imageio' in str(type(item)):
        item = np.array(item)
    elif 'numpy' in str(type(item)):
        pass
    else:  # unsupported type
        print('WTF:', str(type(item)))
        return None
    return item


def metrics(pred, lable):
    # convert to numpy array
    pred = to_numpy(pred)
    lable = to_numpy(lable)

    # convert to 1-D array for convinience
    pred = pred.flatten()
    lable = lable.flatten()
    # convert to 0-1 array
    pred = np.uint8(pred != 0)
    lable = np.uint8(lable != 0)

    met_dict = {}  # metrics dictionary

    TP = np.count_nonzero((pred + lable) == 2)  # true positive
    TN = np.count_nonzero((pred + lable) == 0)  # true negative
    FP = np.count_nonzero(pred > lable)         # false positive
    FN = np.count_nonzero(pred < lable)         # false negative

    smooth = 1e-9                                   # avoid devide zero
    acc = (TP + TN) / (TP + TN + FP + FN + smooth)  # accuracy
    sn = TP / (TP + FN + smooth)                    # sensitivity, or recall
    sp = TN / (TN + FP + smooth)                    # specificity
    pr = TP / (TP + FP + smooth)                    # precession
    f1 = 2 * pr * sn / (pr + sn + smooth)           # F1 mesure
    jac = TP / (TP + FN + FP + smooth)              # jaccard coefficient, IOU
    dice = 2 * TP / (2 * TP + FP + FN + smooth)     # dice coefficient

    # return metrics as dictionary
    met_dict['TP'] = TP
    met_dict['TN'] = TN
    met_dict['FP'] = FP
    met_dict['FN'] = FN
    met_dict['acc'] = acc
    met_dict['sn'] = sn
    met_dict['sp'] = sp
    met_dict['pr'] = pr
    met_dict['f1'] = f1
    met_dict['jac'] = jac
    met_dict['dice'] = dice
    return met_dict


def roc(lable, pred):
    # convert to numpy array
    lable = to_numpy(lable)
    pred = to_numpy(pred)
    # convert to 1-D array for convinience
    pred = pred.flatten()
    lable = lable.flatten()
    # convert lable to 0-1 array
    lable = np.uint8(lable != 0)
    fpr, tpr, thresholds = roc_curve(lable, pred)
    auc_roc = auc(fpr, tpr)  # area under curve
    # plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc.jpg')
    #plt.show()
    return auc_roc
