# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 19:03:01 2023

@author: Rout
"""
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cupy as cp
import scipy
import scipy.sparse as sparse
from scipy.sparse import linalg
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

#print(scipy.show_config())
#print(np.show_config())
try:
    print("number of cuda device(s):",cp.cuda.runtime.getDeviceCount())
except:
    pass
x = cp.array([1, 2, 3])
print("cp device: ", x.device)
# global variables
#This will change the initial condition used. Currently it starts from the first# value
shift_k = 0

approx_res_size = 200


model_params = {'tau': 0.25,
                'nstep': 1000,
                'N': 8,
                'd': 22}

res_params = {'radius':0.1,
             'degree': 3,
             'sigma': 0.5,
             'train_length': 500000,
             'N': int(np.floor(approx_res_size/model_params['N']) * model_params['N']),
             'num_inputs': model_params['N'],
             'predict_length': 2000,
             'beta': 0.0001
              }

# The ESN functions for training
def generate_reservoir(size,radius,degree):
    sparsity = degree/float(size);
    A = sparse.rand(size,size,density=sparsity).todense()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))
    A = (A/e) * radius
    return A

def reservoir_layer(A, Win, input, res_params):
    A = cp.array(A)
    Win = cp.array(Win)
    input = cp.array(input)
    states = cp.zeros((res_params['N'],res_params['train_length']))
    for i in tqdm(range(res_params['train_length']-1)):
        states[:,i+1] = cp.tanh(cp.dot(A,states[:,i]) + cp.dot(Win,input[:,i]))
    return cp.asnumpy(states)


def train_reservoir(res_params, data):
    print("Time:", datetime.now()," Generating reservoir")
    A = generate_reservoir(res_params['N'], res_params['radius'], res_params['degree'])
    q = int(res_params['N']/res_params['num_inputs'])
    Win = cp.zeros((res_params['N'],res_params['num_inputs']))
    for i in tqdm(range(res_params['num_inputs'])):
        cp.random.seed(seed=i)
        Win[i*q: (i+1)*q,i] = res_params['sigma'] * (-1 + 2 * cp.random.rand(1,q)[0])

    print("Time:", datetime.now()," Genrating layer")
    states = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    x = states[:,-1]
    return cp.array(x), Wout, cp.array(A), Win

def train(res_params,states,data):
    beta = res_params['beta']
    idenmat = beta * sparse.identity(res_params['N'])
    states2 = states.copy()
    
    print("Time:", datetime.now()," Calculating U")
    for j in tqdm(range(2,np.shape(states2)[0]-2)):
        if (np.mod(j,2)==0):
            states2[j,:] = (states[j-1,:]*states[j-2,:]).copy()
    U = np.dot(states2,states2.transpose())
    print("U.shape: ", U.shape)
    print("det(A.AT): ", np.linalg.det(U))
    print("rank(A.AT): ", np.linalg.matrix_rank(U))
    U = U + idenmat    # regularization
    print("det(A.AT + Thikonov's regularization): ", np.linalg.det(U))
    print("rank(A.AT + Thikonov's regularization): ", np.linalg.matrix_rank(U))
    
    print("Time:", datetime.now()," Inversion_Starts")
    Uinv = cp.linalg.inv(cp.array(U))
    
    print("Time:", datetime.now()," Calculating Wout")
    data = cp.array(data)
    states2 = cp.array(states2)
    Wout = cp.dot(Uinv,cp.dot(states2,data.transpose()))
    return Wout.transpose()

def predict(A, Win, res_params, x, Wout):
    output = cp.zeros((res_params['num_inputs'],res_params['predict_length']))
    for i in tqdm(range(res_params['predict_length'])):
        x_aug = x.copy()
        for j in range(2,np.shape(x_aug)[0]-2):
            if (np.mod(j,2)==0):
                x_aug[j] = (x[j-1]*x[j-2]).copy()
        out = cp.squeeze(cp.asarray(cp.dot(Wout,x_aug)))
        output[:,i] = out
        x1 = cp.tanh(cp.dot(A,x) + cp.dot(Win,out))
        x = cp.squeeze(cp.asarray(x1))
    return cp.asnumpy(output), cp.asnumpy(x)

print("Time:", datetime.now()," Reading csv")
dataf = pd.read_csv('3tier_lorenz_v3.csv',header=None)

print("Time:", datetime.now()," Transposing")
data = np.transpose(np.array(dataf))

# Train reservoir
x,Wout,A,Win = train_reservoir(res_params,data[:,shift_k:shift_k+res_params['train_length']])

# Prediction
print("Time:", datetime.now(), " Predicting")
output, _ = predict(A, Win,res_params,x,Wout)

print("Time:", datetime.now(), " Saving")
np.save('Expansion_2step_back'+'R_size_train_'+str(res_params['train_length'])+'_Rd_'+str(res_params['radius'])+'_Shift_'+str(shift_k)+'.npy',output)

print("Time:", datetime.now(), " Plotting")
NN = 1000
MTU = 200
xMTU = np.array(range(0, NN))/MTU
var = 3
plt.plot(xMTU, output[var - 1,0:NN], 'r')
plt.plot(xMTU, data[var - 1,499999:499999 + NN], 'k')
plt.margins(x=0)
plt.show()
print("Time:", datetime.now(), " Done")