
import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time

import matplotlib.pyplot as plt
from plotMultivar import plotMultivar

device = torch.device('cpu')
dtype = torch.double
torch.set_default_dtype(dtype)

if dtype == torch.double:
    data = np.loadtxt("/content/drive/MyDrive/UBC Stuff/Summer Research/3tier_lorenz_v3.csv", delimiter=',', dtype=np.float64)
elif dtype == torch.float:
    data = np.loadtxt('/content/drive/MyDrive/UBC Stuff/Summer Research/3tier_lorenz_v3.csv', delimiter=',', dtype=np.float32)

#print(data.shape)
#print("\n")
#print(data)

time_window = 4
n_features = 8
washout = [500]
input_size = time_window*n_features
output_size = n_features
hidden_size = 200
loss_fcn = torch.nn.MSELoss()

train_len = 500000
test_len = 10000

X_data = np.zeros((data.shape[0]-(time_window), time_window, n_features))

#Train Test Split
for i in range (0, data.shape[0]-(time_window)):
    data_slice = data[i:i+time_window]
    X_data[i] = np.expand_dims(data_slice, axis = 0)

X_data = X_data.reshape(X_data.shape[0], X_data.shape[1]*X_data.shape[2])
X_data = np.expand_dims(X_data, axis = 1)
#print(X_data.shape)
#print(X_data)

Y_data = np.expand_dims(data[time_window:, :] - data[time_window-1:-1, :], axis=1)
X_data = torch.from_numpy(X_data).to(device)
Y_data = torch.from_numpy(Y_data).to(device)

#print(X_data.shape)
#print(Y_data.shape)
#print(Y_data)


trX = X_data[:train_len]
trY = Y_data[:train_len]
tsX = X_data[train_len:train_len + test_len]
tsY = Y_data[train_len:train_len + test_len]

#print(trX.shape)
#print(trY.shape)
valdX = tsX.cpu().detach().numpy()
#plotMultivar(valdX, valdX)

#Debug

#print(X_data.shape)
#print("\n")
#print(X_data)


if __name__ == "__main__":
    start = time.time()

    # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

    model = ESN(
        input_size,
        hidden_size,
        output_size,
        nonlinearity="relu",
        spectral_radius= 0.5,
        density = 1,
        lambda_reg= 1,
        leaking_rate = 1
        )
    model.to(device)

    model(trX, washout, None, trY_flat)
    model.fit()
    output, hidden = model(trX, washout)
    print("Training error:", loss_fcn(output, trY[washout[0]:]).item())



    # Test
    #output, hidden = model(tsX, [0], hidden)

    #Rewritten Test Code
    predictedOutput = np.zeros((test_len,8))
    predictedOutput[0:time_window] = data[train_len:train_len+time_window]

    fixedHidden = hidden

    for i in range(0, test_len-time_window):

        step = np.expand_dims([predictedOutput[i:i+time_window].reshape(input_size)], axis = 1)
        step = torch.from_numpy(step).to(device)

        torchOut, hidden = model(step, [0], hidden)
        #print(torchOut.shape)
        predictedOutput[i+time_window] = torchOut.cpu().detach().numpy()[:,0,:] + predictedOutput[i+time_window-1]

    #print("Test error:", loss_fcn(torch.from_numpy(predictedOutput).to(device), tsY).item())
    print("Ended in", time.time() - start, "seconds.")


    #valdX = tsX.detach().numpy()
    #print(f"{output.shape} {valdX.shape}")
    #valdX = valdX[:,0,:]

    plotMultivar(data[train_len+1:train_len+500+1], predictedOutput[:500])
