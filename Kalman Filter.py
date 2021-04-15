import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import filterpy
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, unscented_transform
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter


ball = np.load('../input/trajectories/ball.npy')
attacker = np.load('../input/trajectories/attacker.npy')
defender = np.load('../input/trajectories/defender.npy')




ball_test = ball[28000:,:]
ball = ball[:28000,:]

attacker_test = attacker[28000:,:]
attacker = attacker[:28000,:]

defender_test = defender[28000:,:]
defender = defender[:28000,:]




# model
from filterpy.kalman import KalmanFilter

dt = 1/25
sample = ball[np.random.choice(ball.shape[0])]
x_std = np.std(sample[:,0])
y_std = np.std(sample[:,1])

# dim_x: state vec, dim_z: observation vec
kf = KalmanFilter (dim_x=6, dim_z=2)

kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=max(x_std**2, y_std**2, np.std(sample)**2), block_size=2) # system(process) noise(covariance matrix) in transition state
kf.R = np.diag([x_std**2, y_std**2]) # COVARIANCE of observation noise --> (x,y) + error
# kf.R = np.zeros((2,2))


# states: position, velocity, acceleration
kf.F = np.array([[1, 0, dt, 0, 0.5*dt**2, 0],
                  [0, 1, 0, dt, 0, 0.5*dt**2],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

kf.H = np.array([[1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0]])




# prediction and correction
res = []
for s in sample[:,:]:
     kf.predict()
     kf.update(s)
     res.append(kf.x.copy())
        
result = np.array(res)

plt.figure(figsize=(10,8))
plt.plot(sample[:,0], sample[:,1], color='red', alpha=0.7, label='Ground Truth', linewidth=3)
plt.plot(result[:, 0], result[:, 1], color='green', linestyle='-', alpha=0.7, label='Kalman Filter Prediction', linewidth=3)
plt.legend()
plt.title("Trajectory Prediction: Kalman Filter", fontsize=17)
plt.xlabel("x-position")
plt.ylabel("y-position")

