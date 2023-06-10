
#Libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sklearn

#Curve Fitting
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create an image of the Lorenz attractor.
# The maths behind this code is described in the scipython blog article
# at https://scipython.com/blog/the-lorenz-attractor/
# Christian Hill, January 2016.
# Updated, January 2021 to use scipy.integrate.solve_ivp.

WIDTH, HEIGHT, DPI = 200, 150, 100

# Lorenz paramters and initial conditions.

#BASE : 10, 8/3, 28
#Lorenz Attractor Parameters
sigma, beta, rho = 10, 8/3, 90

#Initial Conditions
u1, v1, w1 = 0, 1, 0

# Maximum time point and total number of time points.
tstart = 0
n = 40000
tmax = 20

#Model Hyperparameters
activ_f = "relu"    #Activation function used
normalize = True    #Normalize/Standardize Data?
alpha_val = 1e-6    #Regularization Strength, more reduces overfitting

#RNG Seeds
seed1 = 1           
seed2 = 42

batch = 200

#Hidden Layers
h_layers = (200,200,200,200,200,200)
iteration = 2000

print(f"Predicting from t = 0s to t = {tmax}s with {n} samples")


#Lorenz Attractor
def lorenz(t, X, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

#generates Coordinates
def generateLorenz(u, v, w):
    # Integrate the Lorenz equations.
    soln = solve_ivp(lorenz, (0, tmax), (u, v, w), args=(sigma, beta, rho),
                    dense_output=True)
    
    # Interpolate solution onto the time grid, t.
    t = np.linspace(tstart, tmax, n)
    x, y, z = soln.sol(t)
    
    coordinates = np.array([t,x,y,z])

    return coordinates.transpose()
 
coordinates = generateLorenz(u1,v1,w1)

#Fits the coordinates with a Neural Network
X = np.array([coordinates[:,1], coordinates[:,2], coordinates[:,3]]).transpose()
t = np.array([coordinates[:, 0]]).transpose()

#Normalize Data
if (normalize == True):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

#Train Test Split
X_train, X_test, t_train, t_test = train_test_split(X, t, random_state= seed1, test_size = 0.3)

#Constructs Model
regr = MLPRegressor(random_state= seed2, max_iter=iteration, hidden_layer_sizes= h_layers, activation = activ_f, alpha = alpha_val, batch_size= batch).fit(t_train, X_train)

#Test score
test_score = regr.score(t_test, X_test)  

#Train_score 
train_score = regr.score(t_train, X_train)

#Predict for all T
X_pred = regr.predict(t)

#Predict only for test set
X_predTest = regr.predict(t_test)
fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection = '3d')

#Plots 3D Graph
plt.figure(1)
ax.plot3D(X_pred[:,0], X_pred[:, 1], X_pred[:, 2])
ax.plot3D(X[:,0], X[:, 1], X[:, 2])
plt.title(f"Trajectory from t = 0 to t = {tmax}\n Test test_score : {test_score}\n Train test_score : {train_score}\n alpha = {alpha_val}\nseed = ({seed1},{seed2})")
#plt.savefig(f"3D Plot, r^2 = {test_score}, actv = {activ_f}.png, norm = {normalize}, a = {alpha_val}, seed = ({seed1},{seed2}).png")

#Plots X vs t
plt.figure(2)
plt.plot(t, X[:,0])
plt.plot(t, X_pred[:,0])
plt.plot(t_test, X_predTest[:,0], "*")
plt.title(f"x vs t from t = {tstart} to t ={tmax}")
#plt.savefig(f"x vs t, r^2 = {test_score}, actv = {activ_f}, norm = {normalize}, a = {alpha_val}, seed = ({seed1},{seed2}).png")

#Plots Y vs t
plt.figure(3)
plt.plot(t, X[:,1])
plt.plot(t, X_pred[:,1])
plt.plot(t_test, X_predTest[:,1], "*")
plt.title(f"y vs t from t = {tstart} to t ={tmax}")
#plt.savefig(f"y vs t, r^2 = {test_score}, actv = {activ_f}, norm = {normalize}, a = {alpha_val}, seed = ({seed1},{seed2}).png")

#Plots Z vs t
plt.figure(4)
plt.plot(t, X[:,2])
plt.plot(t, X_pred[:,2])
plt.plot(t_test, X_predTest[:,2], "*")
plt.title(f"z vs t from t = {tstart} to t ={tmax}")
#plt.savefig(f"z vs t, r^2 = {test_score}, actv = {activ_f}, norm = {normalize}, a = {alpha_val}, seed = ({seed1},{seed2}).png")
plt.show()

print(f"Regression test score : {test_score}")

