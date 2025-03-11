import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

# Define parameters
h = 0.1
nx, ny = 100, 100
dP_dz, mu = -3.2, 1.0e-3
dx = h / (nx - 1)
dy = h / (ny - 1)

# Initialize the w-velocity
w = np.zeros((nx, ny))

# Assemble the system of equations
A = np.zeros((nx * ny, nx * ny))
b = np.zeros(nx * ny)

for i in range(1, nx-1):
    for j in range(1, ny-1):

        west = i - 1 + j * nx
        east = i + 1 + j * nx
        north = i + (j+1) * nx
        south = i + (j-1) * nx
        center = i + j * nx

        aw = mu/dx**2
        ae = mu/dx**2
        an = mu/dy**2
        aso = mu/dy**2
        ac = aw + ae + an + aso

        A[center, west] = -aw
        A[center, east] = -ae
        A[center, north] = -an
        A[center, south] = -aso
        A[center, center] = ac

        b[center] = - dP_dz

# Apply boundary conditions
for i in range(nx):
    for j in [0, ny-1]:
        center = i + j * nx
        A[center, :] = 0
        A[center, center] = 1.0
        b[center] = 0
        
for j in range(ny):
    for i in [0, nx-1]:
        center = i + j * nx
        A[center, :] = 0
        A[center, center] = 1.0
        b[center] = 0

w = solve(A, b)

# Reshape the solution to a 2D array for plotting
w = w.reshape((nx, ny))

# Visualize the velocity distribution
plt.contourf(w, levels=50)
plt.colorbar()
plt.show()