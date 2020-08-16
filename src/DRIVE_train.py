# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 23:12:44 2020

@author: zll
"""
from DRIVE_data_set import DRIVE_data_set
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MF_UNet2 import MF_U_Net
from time import time

#################################
# traing epoches
EPOCHES = 100
# epoches pre-trained
st_ep = 1
# if there is a pre_trained net
PRE_TRAIN = False
# pretrained net path
pretrained_net = 'mf_unet_.pkl'
# batch size
batch_size = 4
#################################

train_set = DRIVE_data_set(train=True)
loader = DataLoader(train_set, batch_size=batch_size)  # data loader
print('data loaded.')


# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# creat and init unet
model = MF_U_Net()
if PRE_TRAIN:
    model.load_state_dict(torch.load(pretrained_net))

model.to(device)  # copy modle to GPU

# loss function
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# training
for epoch in range(st_ep, st_ep + EPOCHES):
    start_time = time()
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        # get input
        inputs, lables = data
        lables = torch.tensor(lables, dtype=torch.long)
        lables = lables.squeeze(1)
        # copy data to GPU
        inputs, lables = inputs.to(device), lables.to(device)
        # set gradiant 0
        optimizer.zero_grad()

        # forwarding
        outputs = model(inputs)
        # compute loss
        loss = criterion(outputs, lables)
        # backwarding
        loss.backward()
        # optimizing
        optimizer.step()

        # print states info
        running_loss += loss.item() * batch_size
    # print epoch loss and time
    print('[%d, loss: %.6f]' % (epoch, running_loss / train_set.len))
    print((time() - start_time) // 60, 'minutes per epoche.')
    # save model every epoch
    torch.save(model.state_dict(), 'mf_unet_'+str(epoch)+'.pkl')
