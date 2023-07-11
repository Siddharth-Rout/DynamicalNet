
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mae_percent
from sklearn.preprocessing import StandardScaler

#Input : <N_samples, N_features>

#Syntax : plotMultivar(vald_traj, pred_traj)
def plotMultivar(vald_traj, pred_traj):

    length = vald_traj.shape[1]
    score = r2_score(vald_traj, pred_traj)
    mae_score = mae_percent(vald_traj, pred_traj)

    #Component Plots
    fig1, axs = plt.subplots(length)

    for i in range(length):

        axs[i].set_ylabel(f"x{i+1}")
        axs[i].plot(vald_traj[:,i])
        axs[i].plot(pred_traj[:,i])

    fig1.suptitle(f"Vector components vs time (in time steps)\nScore: {score}")
    fig1.supxlabel("Number of Timesteps")
    fig1.legend(["actual", "predicted"], loc = "lower right")

    fig1.set_figheight(20)


    #Error
    plt.figure(2)

    for i in range(length):

        plt.plot(np.abs(vald_traj[:,i] - pred_traj[:,i]), label = f"x{i+1} error")

    plt.title(f"Component wise errors\n MAE : {mae_score}")

    plt.figure(3)
    plt.plot(np.abs(vald_traj[:,0] - pred_traj[:,0]))
    plt.title(f"First component eror")

    plt.show()
