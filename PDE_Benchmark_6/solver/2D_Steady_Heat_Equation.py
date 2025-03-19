import numpy as np
import matplotlib.pyplot as plt

# Define the computational domain
x_start, x_end = 0, 5
y_start, y_end = 0, 4

# Define the grid parameters
nx, ny = 101, 81  # number of grid points
dx = (x_end - x_start) / (nx - 1)  # grid size in x direction
dy = (y_end - y_start) / (ny - 1)  # grid size in y direction
beta = dx / dy  # grid aspect ratio

# Initialize the temperature field and set the boundary conditions
T = np.zeros((ny, nx))
T[:, 0] = 10  # left boundary
T[0, :] = 0  # top boundary
T[:, -1] = 40  # right boundary
T[-1, :] = 20  # bottom boundary

# Define the convergence parameters
max_iter = 5000  # maximum number of iterations
tol = 1e-6  # convergence tolerance

# Gauss-Seidel iterative solver
for iter in range(max_iter):
    T_old = T.copy()
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            T[j, i] = (T_old[j, i - 1] + T_old[j, i + 1] + beta**2 * (T_old[j - 1, i] + T_old[j + 1, i])) / (2 * (1 + beta**2))
    if np.abs(T - T_old).max() < tol:
        break

# Save the final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/T_2D_Steady_Heat_Equation.npy', T)

# Visualize the steady-state temperature distribution
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
X, Y = np.meshgrid(x, y)
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, T, levels=50, cmap='jet')
plt.title('Steady-State Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Temperature (°C)')
plt.show()