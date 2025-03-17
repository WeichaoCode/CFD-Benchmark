import numpy as np
import matplotlib.pyplot as plt

# Set the grid parameters
nx, ny = 50, 40  # number of grid points in x and y directions
dx, dy = 5.0/(nx-1), 4.0/(ny-1)  
beta = dx / dy

# Initialize the temperature grid and set the boundary conditions
T = np.zeros((ny, nx))
T[:, 0] = 10.0   # left boundary
T[0, :] = 0.0    # top boundary
T[:, -1] = 40.0  # right boundary
T[-1, :] = 20.0  # bottom boundary

# Set the convergence criterion and maximum number of iterations
eps = 1e-5
max_iter = 500

# Gauss-Seidel iteration
for k in range(max_iter):
    T_old = T.copy()
	
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T[j, i] = (T_old[j, i-1] + T_old[j, i+1] + beta**2 * (T_old[j-1, i] + T_old[j+1, i])) / (2 * (1 + beta**2))

    # Check for convergence
    if np.linalg.norm(T - T_old, np.inf) < eps: 
        break

# Generate the contour plot of the temperature field
plt.figure(figsize=(7,6))
plt.contourf(T, levels=15, cmap=plt.cm.jet)
plt.colorbar(label='Temperature ($^{\circ}$C)')
plt.title('Steady-state Temperature Distribution')

# Save the computed temperature field
np.save('temperature.npy', T)

plt.show()