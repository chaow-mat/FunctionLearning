import os
import numpy as np
import pylab as plt
import sklearn
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from Adam import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import sys

class fcn(nn.Module):
    def __init__(self, in_dim, d_width, c_width, out_dim):
        super(fcn, self).__init__()
        self.in_dim = in_dim
        self.d_width = d_width
        self.c_width = c_width
        self.lift_c = nn.Linear(1, c_width)
        self.lift_d = nn.Linear(in_dim, d_width)
        self.layer1_c = nn.Linear(c_width, c_width)
        self.layer1_d = nn.Linear(d_width, d_width)
        self.layer2_c = nn.Linear(c_width, c_width)
        self.layer2_d = nn.Linear(d_width, d_width)
        self.layer3_c = nn.Linear(c_width, c_width)
        self.layer3_d = nn.Linear(d_width, d_width)
        self.layer4_c = nn.Linear(c_width, 1)
        self.layer4_d = nn.Linear(d_width, out_dim)
        # self.act = nn.gelu()

        self.scale = (1 / (c_width * c_width))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(c_width, c_width, d_width, dtype=torch.float))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(c_width, c_width, d_width, dtype=torch.float))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(c_width, c_width, d_width, dtype=torch.float))



    def forward(self, x):
        b = x.size(0)
        x = x.unsqueeze(2)  # (b, nx, c=1)
        x = self.lift_c(x)  # (b, nx, c=width)
        x = x.permute(0, 2, 1)  # (b, c, nx)
        x = self.lift_d(x)
        x = x.permute(0, 2, 1)  # (b, nx. c)

        x1 = self.layer1_c(x)  # (b, nx, c)
        x2 = torch.einsum("bxi,iox->bxo", x, self.weights1)
        x3 = x.permute(0, 2, 1)  # (b, c, nx)
        x3 = self.layer1_d(x3)
        x3 = x3.permute(0, 2, 1)  # (b, nx, c)
        x = x1 + x2 + x3
        # x = self.act(x)
        x = F.gelu(x)

        x1 = self.layer2_c(x)  # (b, nx, c)
        x2 = torch.einsum("bxi,iox->bxo", x, self.weights2)
        x3 = x.permute(0, 2, 1)  # (b, c, nx)
        x3 = self.layer2_d(x3)
        x3 = x3.permute(0, 2, 1)  # (b, nx, c)
        x = x1 + x2 + x3
        # x = self.act(x)
        x = F.gelu(x)

        x1 = self.layer3_c(x)  # (b, nx, c)
        x2 = torch.einsum("bxi,iox->bxo", x, self.weights3)
        x3 = x.permute(0, 2, 1)  # (b, c, nx)
        x3 = self.layer3_d(x3)
        x3 = x3.permute(0, 2, 1)  # (b, nx, c)
        x = x1 + x2 + x3

        x = self.layer4_c(x)
        x = x.permute(0, 2, 1)  # (b, c, nx)
        x = self.layer4_d(x)

        x = x.squeeze(1)

        return x

# load data
prefix = "/home/wangchao/dataset/FtF/"
data_navier_input = np.load(prefix + "/NavierStokes_inputs.npy")
data_navier_output = np.load(prefix + "/NavierStokes_outputs.npy")

parser = argparse.ArgumentParser()
parser.add_argument('--c_width', type=int, default=16, help='')
parser.add_argument('--d_width', type=int, default=512)
parser.add_argument('--M',  type=int, default=2500, help="number of dataset")
parser.add_argument('--dim_PCA', type=int, default=200)
parser.add_argument('--device', type=int, default=0, help="index of cuda device")
cfg = parser.parse_args()

print(sys.argv)
print('N training: ', cfg.M, ' N testing : ', cfg.M, ' d_width: ', cfg.d_width, ' c_width: ', cfg.c_width,' dim_PCA: ', cfg.dim_PCA)
# whiten the data
data_navier_input = data_navier_input - np.mean(data_navier_input, axis=2)[:, :, None]
data_navier_output = data_navier_output - np.mean(data_navier_output, axis=2)[:, :, None]

n_train = cfg.M  # N
n_test = cfg.M
X_train = data_navier_input[:, :, :n_train]
X_test = data_navier_input[:, :, n_train:n_train+n_test]
Y_train = data_navier_output[:, :, :n_train]
Y_test = data_navier_output[:, :, n_train:n_train+n_test]
X_train_flat = X_train.reshape(64 * 64, -1)  # X
Y_train_flat = Y_train.reshape(64 * 64, -1)  # Y
X_test_flat = X_test.reshape(64 * 64, -1)  # \tilde{X}
Y_test_flat = Y_test.reshape(64 * 64, -1)  # \tilde{Y}


# compute covariance matrix
# first, flatten the data
X_covariance = X_train_flat @ X_train_flat.T
Y_covariance = Y_train_flat @ Y_train_flat.T

# Compute PCA componebts

# compute the spectrum of the covariance matrix
cov_X_eigenval, cov_X_eigenvec = np.linalg.eigh(X_covariance)
cov_Y_eigenval, cov_Y_eigenvec = np.linalg.eigh(Y_covariance)

# largest eigenvalues at the start (numpy convention is to place them at the end)
X_PCs = cov_X_eigenvec[:, ::-1]
Y_PCs = cov_Y_eigenvec[:, ::-1]

# training data
dim_PCA = cfg.dim_PCA
X_PCs_small = X_PCs[:, :dim_PCA]
Y_PCs_small = Y_PCs[:, :dim_PCA]
X_coeff = X_train_flat.T @ X_PCs_small
Y_coeff = Y_train_flat.T @ Y_PCs_small
X_coeff_test = X_test_flat.T @ X_PCs_small
Y_coeff_test = Y_test_flat.T @ Y_PCs_small
norms = np.mean((Y_test_flat.T) ** 2, axis=1)

device = cfg.device
layer = 3
c_width = cfg.c_width
d_width = cfg.d_width
batch_size = 64
learning_rate = 0.001
num_epoches = 10000
ep_predict = 10
step_size = 500
gamma = 0.5


train_i = torch.from_numpy(X_coeff).to(torch.float32)
train_o = torch.from_numpy(Y_coeff).to(torch.float32)
test_i = torch.from_numpy(X_coeff_test).to(torch.float32)
test_o = torch.from_numpy(Y_coeff_test).to(torch.float32)

# cp to device
Y_PCs_small = torch.from_numpy(Y_PCs_small.copy()).to(device).to(torch.float32)
Y_test_flat = torch.from_numpy(Y_test_flat.copy()).to(device).to(torch.float32)

device = torch.device('cuda:' + str(device))
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_i, train_o), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_i, test_o, Y_test_flat.T), batch_size=batch_size, shuffle=False)


model = fcn(dim_PCA, d_width, c_width, dim_PCA)
# print(count_params(model))
model = model.float()
# print(count_params(model))

if torch.cuda.is_available():
    model = model.to(device)



criterion = nn.MSELoss(reduction='sum')  # Loss
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# schedule_PCA_ad = 3
# milestones = [200, 500]
# scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

path = './ns/PCA-new-net-12/'
path = path + 'N_' + str(n_train) + '_dpca_' + str(dim_PCA) + '_l_' + str(layer) + '_act_gelu_' + '_dw_' + str(d_width) + '_cw_' + str(c_width) +'_lr_' + str(learning_rate) + '-' + str(step_size) + '-' + str(gamma) + '_bs_' + str(batch_size)
if not os.path.exists(path):
        os.makedirs(path)
writer = SummaryWriter(log_dir=path)

for ep in range(num_epoches):
    model.train()
    train_l2_step = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        train_l2_step += loss.item()

        #  backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    test_l2_step = 0
    average_relative_error = 0
    average_relative_error2 = 0
    with torch.no_grad():
        for x, y, Y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            test_l2_step += loss.item()
            if ep % ep_predict == 0:
                Y_test_pred = out @ Y_PCs_small.T
                norms = torch.mean(Y ** 2, axis=1)
                norms2 = torch.norm(Y, dim=1)
                error = Y_test_pred - Y
                relative_error = torch.mean(error ** 2, axis=1) / norms
                relative_error2 = torch.norm(error, dim=1) / norms2
                average_relative_error += torch.sum(relative_error)
                average_relative_error2 += torch.sum(relative_error2)
    # print('Epoch[{}/{}], loss: {:.6e} / {:.6e}'.format(ep + 1, num_epoches, train_l2_step / n_train, test_l2_step / n_train))
    writer.add_scalar("train/loss", train_l2_step / n_train, ep)
    writer.add_scalar("test/loss", test_l2_step / n_train, ep)
    if ep % ep_predict == 0:
        average_relative_error = average_relative_error/(n_test)
        average_relative_error2 = average_relative_error2 / (n_test)
        print(f"Average Relative Error of original PCA: {average_relative_error: .6e} {average_relative_error2: .6e}")
        writer.add_scalar("test/error", average_relative_error, ep)
        writer.add_scalar("test2/error", average_relative_error2, ep)



