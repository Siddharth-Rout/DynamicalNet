import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Lorenz paramters and initial conditions.
sigma, beta, rho = 10, 8/3, 28
u0, v0, w0 = 0, 1, 0

# Maximum time point and total number of time points.
time_interval = 0.001
tmax = 20
n = int(tmax/ time_interval)

def lorenz(t, X, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

# Integrate the Lorenz equations.
soln = solve_ivp(lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho),
                dense_output=True)

# Interpolate solution onto the time grid, t.
t = np.linspace(0, tmax, n)
x, y, z = soln.sol(t)

data = {'X': x, 'Y':y, 'Z':z, 't':t}
df = pd.DataFrame(data)
df.to_excel('Data.xlsx', index = False)

# Load the data from the Excel file
df = pd.read_excel('Data.xlsx')
t = df[['t']]
X = df[['X', 'Y', 'Z']]

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the data into training and test sets
t_train, t_test, X_train, X_test = train_test_split(t, X_normalized, test_size=0.1, random_state=1)
step = 1

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(units = 32, input_shape=(X_train.shape[1], step), activation = "relu"))
model.add(Dense(3))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

# Train the model (Autoencoder) - The input values are used as output
model.fit(X_train, X_train, epochs=100, batch_size=32)

# Make predictions on the train and test set
X_pred = model.predict(X_test)

# test score (Autoencoder)
score = model.evaluate(X_test, X_test, verbose = 0)
score_percent = score * 100

# Inverse transform the predicted data
X_pred = scaler.inverse_transform(X_pred)

# Plotting the results
X = np.array(X)

#3d Plotting
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
X_test = np.array(X_test)
ax.plot3D(X[:,0], X[:,1], X[:,2],'.', label = "Actual")
ax.set_title((f"Score=%.3f" %score_percent) + '%')
ax.set_xlabel("X", color = "red")
ax.set_ylabel("Y", color = "red")
ax.set_zlabel("Z", color = "red")

X_pred = np.array(X_pred)
ax.plot3D(X_pred[:,0], X_pred[:,1], X_pred[:,2],'*', label = "Predicted")
ax.legend()

#Plotting X vs t
plt.figure(3)


plt.plot(t,X[:,0], ".b", label = "Actual")
plt.plot(t_test,X_pred[:,0], ".r", label = "Predicted")
# plt.plot(t,X[:,0], color ="green")
plt.title("Comparison of Predicted X and X test")

plt.xlabel("time")
plt.ylabel("X")
plt.legend()

#Plotting y vs t
plt.figure(4)

plt.plot(t,X[:,1], ".b", label = "Actual")
plt.plot(t_test,X_pred[:,1], ".r", label = "Predicted")
# plt.plot(t,X[:,1], color ="green")
plt.title("Comparison of Predicted Y and Y test")

plt.xlabel("time")
plt.ylabel("Y")
plt.legend()


#Plotting z vs t
plt.figure(5)

plt.plot(t,X[:,2], ".b", label = "Actual")
plt.plot(t_test,X_pred[:,2], ".r", label = "Predicted")
# plt.plot(t,X[:,2], color ="green")
plt.title("Comparison of Predicted Z and Z test")

plt.xlabel("time")
plt.ylabel("Z")
plt.legend()
 
plt.show()



