"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import os
import sys
sys.path.append('nn')
from mynn import *
from mydata import UnitGaussianNormalizer
from Adam import Adam
from tensorboardX import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import argparse


# torch.manual_seed(0)
# np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--modes', type=int, default=12, help='')
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--M',  type=int, default=2500, help="number of dataset")
parser.add_argument('--device', type=int, default=1, help="index of cuda device")
cfg = parser.parse_args()



################################################################
# load data and data normalization
################################################################
print(sys.argv)
print('N training: ', cfg.M, ' N testing: ', cfg.M, ' width: ', cfg.width, ' modes: ', cfg.modes)


batch_size = 64
M = cfg.M
width = cfg.width
modes = cfg.modes




s = N = 41 # dimension of data
ntrain = M
ntest = M
N_theta = 100

device = cfg.device
device = torch.device('cuda:' + str(device))

prefix = "/home/wangchao/dataset/FtF/"
cs = np.load(prefix + "/StructuralMechanics_inputs.npy")
K = np.load(prefix + "/StructuralMechanics_outputs.npy")

xgrid = np.linspace(0,1,N)
dx    = xgrid[1] - xgrid[0]
Y, X = np.meshgrid(xgrid, xgrid)
# test
i = 20
j = 40
assert(X[i, j] == i*dx and Y[i, j] == j*dx)

# transpose
cs = cs.transpose(2, 0, 1)
K = K.transpose(2, 0, 1)


x_train = torch.from_numpy(np.reshape(cs[:ntrain, :, :], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(K[:ntrain, :, :], -1).astype(np.float32))

x_test = torch.from_numpy(np.reshape(cs[ntrain:ntrain+ntest, :,  :], -1).astype(np.float32))
y_test = torch.from_numpy(np.reshape(K[ntrain:ntrain+ntest, :,  :], -1).astype(np.float32))


x_normalizer = UnitGaussianNormalizer(x_train, device)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train, device)
y_train = y_normalizer.encode(y_train)


x_train = x_train.reshape(ntrain,s, s ,1)
x_test = x_test.reshape(ntest,s, s, 1)

# todo do we need this
y_train = y_train.reshape(ntrain, s, s, 1)
y_test = y_test.reshape(ntest, s, s, 1)



################################################################
# training and evaluation
################################################################

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

learning_rate = 0.001

epochs = 5000
step_size = 100
gamma = 0.5

# modes = 12

path = "sm/FNO/FNO_sm_"+str(width)+"Nd_"+str(ntrain)+"m_" +str(modes) + '_lr_' + str(learning_rate) + '-' + str(step_size) + '-' + str(gamma)
if not os.path.exists(path):
        os.makedirs(path)
writer = SummaryWriter(log_dir=path)


model = FNO2d(modes, modes, width).to(device)
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
t0 = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        batch_size_ = x.shape[0]
        optimizer.zero_grad()
        out = model(x).reshape(batch_size_, s,  s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    # torch.save(model, "FNO_"+str(width)+"Nd_"+str(ntrain)+".model")
    scheduler.step()

    train_l2/= ntrain

    t2 = default_timer()
    writer.add_scalar("train/error", train_l2, ep)
    # print("Epoch : ", ep, " Epoch time : ", t2-t1, " Rel. Train L2 Loss : ", train_l2)

    average_relative_error = 0
    average_relative_error2 = 0
    with torch.no_grad():
        for x, y, in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size_ = x.shape[0]
            out = model(x).reshape(batch_size_, s, s)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            if ep % 10 == 0:
                out = out.reshape(batch_size_, -1)
                y = y.reshape(batch_size_, -1)
                norms = torch.mean(out ** 2, axis=1)
                norms2 = torch.norm(out, dim=1)
                error = y - out
                relative_error = torch.mean(error ** 2, axis=1) / norms
                relative_error2 = torch.norm(error, dim=1) / norms2
                average_relative_error += torch.sum(relative_error)
                average_relative_error2 += torch.sum(relative_error2)
    if ep % 10 == 0:
        average_relative_error = average_relative_error / (ntest)
        average_relative_error2 = average_relative_error2 / (ntest)
        print(f"Average Relative Test Error: {average_relative_error: .6e} {average_relative_error2: .6e}")
        writer.add_scalar("test/error", average_relative_error, ep)
        writer.add_scalar("test2/error", average_relative_error2, ep)

# print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)
