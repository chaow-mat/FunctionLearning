import os
import numpy as np
import pylab as plt
import sklearn
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from Adam import Adam
import argparse

class fcn(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(fcn, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden), nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden, out_dim))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
parser = argparse.ArgumentParser()
# parser.add_argument('--modes', type=int, default=12, help='')
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--M',  type=int, default=2500, help="number of dataset")
parser.add_argument('--device', type=int)
parser.add_argument('--dim_PCA', type=int, default=100)
cfg = parser.parse_args()
# print(cfg)
print('N training: ', cfg.M, ' N testing: ', cfg.M, ' width: ', cfg.width, ' dim_PCA: ', cfg.dim_PCA)

# load data
prefix = "/home/wangchao/dataset/FtF/"
data_navier_input = np.load(prefix + "/NavierStokes_inputs.npy")
data_navier_output = np.load(prefix + "/NavierStokes_outputs.npy")

# whiten the data
data_navier_input = data_navier_input - np.mean(data_navier_input, axis=2)[:, :, None]
data_navier_output = data_navier_output - np.mean(data_navier_output, axis=2)[:, :, None]

n_train = cfg.M  # N
# n_test = 40000 - n_train
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
width = cfg.width
batch_size = 32
learning_rate = 1e-3
num_epoches = 5000
ep_predict = 10
step_size = 500
gamma = 0.5

# X_coeff = np.random.uniform(0, 2*np.pi, (n_train, dim_PCA))
# Y_coeff = np.random.uniform(0, 2*np.pi, (n_train, dim_PCA))
# X_coeff_test = np.random.uniform(0, 2*np.pi, (n_train, dim_PCA))
# Y_coeff_test = np.random.uniform(0, 2*np.pi, (n_train, dim_PCA))

train_i = torch.from_numpy(X_coeff).to(torch.float32)
train_o = torch.from_numpy(Y_coeff).to(torch.float32)
test_i = torch.from_numpy(X_coeff_test).to(torch.float32)
test_o = torch.from_numpy(Y_coeff_test).to(torch.float32)

# cp to device
Y_PCs_small = torch.from_numpy(Y_PCs_small.copy()).to(device).to(torch.float32)
Y_test_flat = torch.from_numpy(Y_test_flat.copy()).to(device).to(torch.float32)
# norms = torch.from_numpy(norms).to(device).to(torch.float32)

device = torch.device('cuda:' + str(device))
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_i, train_o), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_i, test_o, Y_test_flat.T), batch_size=batch_size, shuffle=False)
path = '/home/wangchao/Codes/PCA-nn/results/PCA-net/ns/'
path = path + 'N_' + str(n_train) + '_dpca_' + str(dim_PCA) + '_w_' + str(width) + '_lr_' + str(learning_rate) + '_bs_' + str(batch_size)
if not os.path.exists(path):
        os.makedirs(path)
writer = SummaryWriter(log_dir=path)

model = fcn(dim_PCA, width, dim_PCA)
model = model.float()
# print(count_params(model))

if torch.cuda.is_available():
    model = model.to(device)



criterion = nn.MSELoss(reduction='sum')  # Loss
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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
        # print(f"Average Relative Error of original PCA: {average_relative_error: .6e}")
        writer.add_scalar("test/error", average_relative_error, ep)
        writer.add_scalar("test2/error", average_relative_error2, ep)



