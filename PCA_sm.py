import os
import sys
import numpy as np

sys.path.append('nn')
from mynn import *
from mydata import UnitGaussianNormalizer
from Adam import Adam
from timeit import default_timer
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
# parser.add_argument('--modes', type=int, default=12, help='')
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--M', type=int, default=2500, help="number of dataset")
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--dim_PCA', type=int, default=200)
cfg = parser.parse_args()
# print(cfg)
print('N training: ', cfg.M, ' N testing: ', cfg.M, ' width: ', cfg.width, ' dim_PCA: ', cfg.dim_PCA)

device = torch.device('cuda:' + str(cfg.device))

M = cfg.M
N_neurons = cfg.width
layers = 3
batch_size = 64

N = 21
ntrain = M
ntest = M
N_theta = 100
# load data
prefix = "/home/wangchao/dataset/FtF/"
data_sm_input = np.load(prefix + "/StructuralMechanics_inputs.npy")
data_sm_output = np.load(prefix + "/StructuralMechanics_outputs.npy")

acc = 0.999


inputs = data_sm_input
outputs = data_sm_output

compute_input_PCA = True

if compute_input_PCA:
    train_inputs = np.reshape(inputs[:, :, :ntrain], (-1, ntrain))
    test_inputs = np.reshape(inputs[:, :, ntrain:ntrain+ntest], (-1, ntest))
    Ui, Si, Vi = np.linalg.svd(train_inputs)
    en_f = 1 - np.cumsum(Si) / np.sum(Si)
    r_f = np.argwhere(en_f < (1 - acc))[0, 0]

    r_f = cfg.dim_PCA
    Uf = Ui[:, :r_f]
    f_hat = np.matmul(Uf.T, train_inputs)
    f_hat_test = np.matmul(Uf.T, test_inputs)
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))
    x_test = torch.from_numpy(f_hat_test.T.astype(np.float32))
else:

    print("must compute input PCA")

train_outputs = np.reshape(outputs[:, :, :ntrain], (-1, ntrain))
test_outputs = np.reshape(outputs[:, :, ntrain:ntrain+ntest], (-1, ntest))
Uo, So, Vo = np.linalg.svd(train_outputs)
en_g = 1 - np.cumsum(So) / np.sum(So)
r_g = np.argwhere(en_g < (1 - acc))[0, 0]
r_g = cfg.dim_PCA
Ug = Uo[:, :r_g]
g_hat = np.matmul(Ug.T, train_outputs)
g_hat_test = np.matmul(Ug.T, test_outputs)
y_train = torch.from_numpy(g_hat.T.astype(np.float32))
y_test = torch.from_numpy(g_hat_test.T.astype(np.float32))
test_outputs = torch.from_numpy(test_outputs).to(device)

x_normalizer = UnitGaussianNormalizer(x_train, device)
x_train = x_normalizer.encode(x_train)
y_normalizer = UnitGaussianNormalizer(y_train, device)
y_train = y_normalizer.encode(y_train)

print("Input #bases : ", r_f, " output #bases : ", r_g)

################################################################
# training and evaluation
################################################################


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, test_outputs.T),
                                          batch_size=batch_size, shuffle=False)
learning_rate = 0.0001
epochs = 5000
step_size = 100
gamma = 0.5

path = "sm/PCA/PCA_sm_dpca" +str(cfg.dim_PCA) + '_cw'+ str(cfg.width) + "Nd_" + str(ntrain) + '_lr_' + str(learning_rate) + '-' + str(
    step_size) + '-' + str(gamma)
if not os.path.exists(path):
    os.makedirs(path)
writer = SummaryWriter(log_dir=path)

model = FNN(r_f, r_g, layers, N_neurons)
print(count_params(model))
model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = torch.nn.MSELoss(reduction='sum')
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
        out = model(x)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out, y)
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    # torch.save(model, "PCANet_" + str(N_neurons) + "_" + str(layers) + "Nd_" + str(ntrain) + ".model")
    scheduler.step()

    train_l2 /= ntrain

    t2 = default_timer()
    writer.add_scalar("train/loss", train_l2, ep)
    # print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)

    average_relative_error = 0
    average_relative_error2 = 0
    with torch.no_grad():
        for x, y, y_test in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size_ = x.shape[0]
            out = model(x)
            out = y_normalizer.decode(out).detach().cpu().numpy()
            y = y_normalizer.decode(y)
            if ep % 10 == 0:
                y_test = y_test.detach().cpu().numpy()
                y_test_pred = np.matmul(Ug, out.T)
                norms = np.mean(y_test ** 2, axis=1)
                norms2 = np.linalg.norm(y_test, axis=1)
                error = y_test - y_test_pred.T
                relative_error = np.mean(error ** 2, axis=1) / norms
                relative_error2 = np.linalg.norm(error, axis=1) / norms2
                average_relative_error += np.sum(relative_error)
                average_relative_error2 += np.sum(relative_error2)
    if ep % 10 == 0:
        average_relative_error = average_relative_error / (ntest)
        average_relative_error2 = average_relative_error2 / (ntest)
        print(f"Average Relative Test Error of PCA: {average_relative_error: .6e} {average_relative_error2: .6e}")
        writer.add_scalar("test/error", average_relative_error, ep)
        writer.add_scalar("test2/error", average_relative_error2, ep)

# print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)

