import numpy as np
import matplotlib.pyplot as plt

# Define parameters
H = 1.0    # channel height
ny = 101   # number of grid points
mu = 1.0e-3   # kinematic viscosity
mut = lambda y: 0.01 * (1 - 4*y*(1-y)/H**2)  # eddy viscosity profile
dy = H / (ny - 1)  # grid spacing

# Discretize the domain
y = np.linspace(0, H, ny)

# Assemble the coefficient matrix A and RHS vector b
A = np.zeros((ny, ny))
b = -np.ones(ny)  # constant pressure gradient

for i in range(1, ny-1):
  A[i, i-1] = -(mu + mut(y[i-1])) / dy**2
  A[i, i] = ((mu + mut(y[i-1])) + (mu + mut(y[i+1]))) / dy**2
  A[i, i+1] = -(mu + mut(y[i+1])) / dy**2

# Enforce boundary conditions
A[0, 0] = A[-1, -1] = 1.0
b[0] = b[-1] = 0.0

# Solve for velocity u
u = np.linalg.solve(A, b)

# Visualize the velocity profile
plt.figure(figsize=(6, 8))
plt.plot(u, y)
plt.xlabel('Velocity u')
plt.ylabel('Height y')
plt.grid()
plt.show()