import numpy as np
import matplotlib.pyplot as plt

# Define constants
mu = 1e-3  # dynamic viscosity
dpdz = -3.2  # pressure gradient
h = 0.1  # duct side length
nx = ny = 80  # number of grid points in x and y direction

# Define grid spacing
dx = dy = h / (nx - 1)

# Initialize velocity field
w = np.zeros((nx, ny))

# Define coefficient matrix
aE = aW = mu * dy / dx
aN = aS = mu * dx / dy
aP = aE + aW + aN + aS
Su = dpdz * dx * dy

# Jacobi iteration solver
for iter in range(5000):
    w_old = w.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            w[i, j] = (aE * w_old[i+1, j] + aW * w_old[i-1, j]
                       + aN * w_old[i, j+1] + aS * w_old[i, j-1] - Su) / aP

# Save the velocity field
np.save('velocity_field.npy', w)

# Visualize the solution
plt.figure(figsize=(8, 6))
plt.contourf(w, cmap='viridis')
plt.title('Velocity distribution in the square duct')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='w')
plt.show()