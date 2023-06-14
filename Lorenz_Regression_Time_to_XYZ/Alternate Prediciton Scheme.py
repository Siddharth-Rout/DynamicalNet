
#Libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sklearn


#Curve Fitting
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split as shuffle
from sklearn.preprocessing import StandardScaler

# Lorenz paramters and initial conditions.

#BASE : 10, 8/3, 28
#Lorenz Attractor Parameters
sigma, beta, rho = 10, 8/3, 28

#Initial Conditions
u1, v1, w1 = 1, 1, 1

# Maximum time point and total number of time points.
tstart = 0
dt = 0.01
tmax = 10**3
N = int((tmax-tstart)/dt)

#ResNet parmaeters
n_M = 0   #Number of memory parameters
dimSetup = [1, 1, 1]
#[X, Y, Z]

#Model Hyperparameters
activ_f = "relu"    #Activation function used
alpha_val = 1e-2    #Regularization Strength, more reduces overfitting
learn_rate = "adaptive"
early_stop = False

#RNG Seeds
seed1 = 1           
seed2 = 42

batch = 400

#Hidden Layers
h_layers = (800,800,600,600,400,400,200,200)
iteration = 200

print(f"Predicting from t = 0s to t = {tmax}s with {N} samples")

#Lorenz Attractor
def lorenz(t, X, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

#generates Coordinates
def generateLorenzCoordinates(u, v, w):
    # Integrate the Lorenz equations.
    soln = solve_ivp(lorenz, (0, tmax), (u, v, w), args=(sigma, beta, rho),
                    dense_output=True)
    
    # Interpolate solution onto the time grid, t.
    t = np.linspace(tstart, tmax, N)
    x, y, z = soln.sol(t)
    x = x*dimSetup[0]
    y = y*dimSetup[1]
    z = z*dimSetup[2]
    
    coordinates = np.array([t,x,y,z])

    return coordinates.transpose()

def constructTrainTestDatasets(coordinates, n_Mem):

    X = np.array([coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]]).transpose()
    #Input Data : X_n + memory terms
    #Test Data : X_n+1

    print(X)
    print(np.shape(X))

    if(n_Mem != 0):
        X_input = [X[0 : n_Mem+1].flatten()]
        for i in range(n_Mem+1, N-1):
            temp_array = [X[i-n_Mem : i+1].flatten()]
            X_input = np.concatenate((X_input, temp_array), axis = 0)

    else : X_input = X[0:N-1]

    X_target = X[n_Mem+1:N]
    dX_target = X[n_Mem+1:N] - X[n_Mem: N-1]
    return X_target, dX_target, X_input

#Neural Network
#Train Test Split

def activateNeuralNetwork(X_in, dX):

    Xi_train, Xi_test, dXt_train, dXt_test = shuffle(X_in, dX, random_state = seed1, test_size = 0.1, shuffle= False)

    NN = MLPRegressor(

                    random_state= seed2, 
                    max_iter=iteration, 
                    hidden_layer_sizes= h_layers, 
                    activation = activ_f, 
                    alpha = alpha_val, 
                    batch_size= batch, 
                    learning_rate= learn_rate,
                    early_stopping= early_stop,
                    verbose= True,
                    n_iter_no_change= 50
                    
                    ).fit(Xi_train, dXt_train)

    print(NN.score(Xi_train, dXt_train))
    print(NN.score(Xi_test, dXt_test))

    return NN

#NO RESNET VER
def predictNextStep(trajectory, X_input, n_steps):
    
    a = np.arange(0,n_steps-1)
    for i in np.nditer(a):
        X_input = NN.predict(X_input) + X_input
        trajectory = np.concatenate((trajectory, X_input), axis = 0)

    return trajectory       

#WITH RESNET VER

def RN_predict(trajectory, X_inputs, n_steps, n_mem):

    #For large N, should be fine
    #generate first n_mem steps without memory

    a = np.arange(0,n_steps-n_mem-1)
    
    for i in np.nditer(a):
        X_cur = X_inputs[n_mem]
        X_inputs = [X_inputs.flatten()]

        X_cur = NN.predict(X_inputs) + X_cur
        trajectory = np.concatenate((trajectory, X_cur), axis = 0)
        X_inputs = trajectory[i+1 : n_mem+i+2]

    return trajectory   


#MAIN
coords = generateLorenzCoordinates(u1,v1,w1)

X_target, dX_target, X_input = constructTrainTestDatasets(coords, n_M)

NN = activateNeuralNetwork(X_input, dX_target)

X_initial = np.array(coords[0 : n_M+1 , 1:4])
trajectory = X_initial

if (n_M == 0):
    pred_traj = predictNextStep(trajectory, X_initial, N)

else : 
    pred_traj = RN_predict(trajectory, X_initial, N, n_M)

t = np.linspace(tstart+dt, tmax, N)

score = sklearn.metrics.r2_score(pred_traj, coords[:,1:4])
print(f"r^2 value {score}")

#Plotting
fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection = '3d')

plt.figure(1)
ax.plot3D(pred_traj[:,0], pred_traj[:, 1], pred_traj[:, 2])
ax.plot3D(coords[:,1], coords[:, 2], coords[:, 3])
plt.legend(["predicted", "actual"], loc = "lower right")

plt.figure(2)
plt.plot(pred_traj[:,0], "x")
plt.plot(coords[:,1])
plt.xlabel("time steps")
plt.ylabel("x")
plt.title("X vs time steps")
plt.legend(["predicted", "actual"], loc = "lower right")

plt.figure(3)
plt.plot(pred_traj[:,1], "x")
plt.plot(coords[:,2])
plt.title("Y vs time steps")
plt.xlabel("time steps")
plt.ylabel("y")
plt.legend(["predicted", "actual"], loc = "lower right")

plt.figure(4)
plt.plot(pred_traj[:,2], "x")
plt.plot(coords[:,3])
plt.title("Z vs time steps")
plt.xlabel("time steps")
plt.ylabel("z")
plt.legend(["predicted", "actual"], loc = "lower right")

plt.figure(5)
plt.plot(np.abs(pred_traj[:,0] - coords[:,1]))
plt.plot(np.abs(pred_traj[:,1] - coords[:,2]))
plt.plot(np.abs(pred_traj[:,2] - coords[:,3]))
plt.title("Component wise errors")
plt.legend(["x error", "y error", "z error"], loc = "lower right")

plt.show()

#Need to Move to Jupyter for now
