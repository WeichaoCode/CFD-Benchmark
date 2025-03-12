import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D

# Define parameters
L = 2
T_cold = 0
T_hot = 200
alpha = 1e-4
sigma = 0.1
Nx = 50
Ny = 50
Nt = 5000
dx = L / (Nx - 1)
dy = L / (Ny - 1)
dt = (L / Nx) ** 2 / (4 * alpha)

# Initialize grid.
T = np.empty((Nx, Ny, 2), dtype=np.float64)
T.fill(T_cold)

# apply initial conditions
X, Y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))
Q = T_hot * np.exp(-(X**2 + Y**2)/(2*sigma**2))
T[:, :, 0] += Q
T[:, :, 1] += Q

# time evolve
for t in range(2, Nt + 2):
    T[:, :, t%2] = ((T[:, :, (t-1)%2] + alpha*dt/dx**2 *
                    (np.roll(T[:, :, (t-2)%2], 1, axis=1) - 
                     2*T[:, :, (t-2)%2] +
                     np.roll(T[:, :, (t-2)%2], -1, axis=1)) +
                     alpha*dt/dy**2 *
                     (np.roll(T[:, :, (t-2)%2], 1, axis=0) -
                      2*T[:, :, (t-2)%2] +
                      np.roll(T[:, :, (t-2)%2], -1, axis=0))) /
                     (1 + 2*alpha*dt*(1/dx**2 + 1/dy**2)))