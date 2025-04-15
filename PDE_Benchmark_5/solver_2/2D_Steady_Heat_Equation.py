import numpy as np
import matplotlib.pyplot as plt

# Initialize the number of iterations and convergence criterion
maxIter = 500
tol = 1.0e-5

# Initialize the grid 
lx = 5
ly = 4
nx = 25
ny = 20

dx = lx / (nx - 1)
dy = ly / (ny - 1)
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
T = np.zeros((ny, nx))
T_new = np.zeros((ny, nx))

# Boundary conditions
T[:, 0] = 10  # Left edge temperature
T[0, :] = 20  # Bottom edge temperature
T[-1, :] = 0  # Top edge temperature
T[:, -1] = 40  # Right edge temperature

# Gauss-Seidel iteration
for iteration in range(maxIter):
    # Copy temperature from previous iteration
    T_new = np.copy(T)
    # Loop over grid points
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            T_new[j, i] = (dy ** 2 * (T[j, i - 1] + T[j, i + 1]) + dx ** 2 * (T[j - 1, i] + T[j + 1, i])) / (
                        2 * (dx ** 2 + dy ** 2))

    # Check if we reached the desired precision level
    if np.abs(T_new - T).max() < tol:
        break

    T = T_new

# Plotting
plt.contourf(x, y, T, levels=50, cmap='hot', origin='lower')
plt.title('$T(x,y)$ Steady State Heat Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Temperature')
plt.show()