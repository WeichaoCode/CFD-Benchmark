# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the size of the grid
N = 50 
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

# Define time parameters and thermal diffusivity
dt = 0.001
t_final = 0.1
alpha = 0.1
t_grid = np.arange(0, t_final, dt)
Q_0 = 200
sigma = 0.1

# Initialize temperature and heat source
T = np.zeros((N, N, len(t_grid)))
q = Q_0 * np.exp( - (X**2 + Y**2) / (2*sigma**2) )

# Set boundary conditions
T[:,:,0] = 0
T[0,:,:] = 0
T[N-1,:,:] = 0
T[:,0,:] = 0
T[:,N-1,:] = 0

# Explicit Scheme
for t in range(len(t_grid)-1):
    T[:,:,t+1] = T[:,:,t] + dt * ( alpha * ( np.roll(T[:,:,t], -1, axis=0) - 2 * T[:,:,t] 
        + np.roll(T[:,:,t], 1, axis=0) + np.roll(T[:,:,t], -1, axis=1) - 2 * T[:,:,t] + np.roll(T[:,:,t], 1, axis=1) ) + q )

# This is where you would implement the Implicit Scheme
# It involves solving a system of linear equations at each timestep, commonly done with scipy.linalg's solve methods

# Visualizing the results using a 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, T[:,:,-1])

plt.show()