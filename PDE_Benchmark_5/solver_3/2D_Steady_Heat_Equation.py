import numpy as np
import matplotlib.pyplot as plt

# Parameters
lx, ly = 5, 4  # domain dimensions
nx, ny = 101, 101  # number of points (including boundary)
dx, dy = lx / (nx - 1), ly / (ny - 1)  # grid spacing
T_left, T_right, T_top, T_bottom = 10, 40, 0, 20  # boundary conditions
T_guess = 30  # initial guess for T

# Initialize T array
T = np.full((ny, nx), T_guess)

# Set boundary conditions
T[:, 0] = T_left
T[:, -1] = T_right
T[0, :] = T_top
T[-1, :] = T_bottom

# Gauss-Seidel method
iter_max = 10000  # maximum number of iterations
T_tol = 1e-6  # convergence criterion (temperature change)
for it in range(iter_max):
    T_old = T.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            T[j, i] = 0.5 * ((T_old[j, i-1] + T_old[j, i+1]) / dx**2 +
                              (T_old[j-1, i] + T_old[j+1, i]) / dy**2) /\
                       (1/dx**2 + 1/dy**2)
    if np.abs(T - T_old).max() < T_tol:
        print("Converged in", it, "iterations.")
        break
else:
    print("Didn't reach convergence in", iter_max, "iterations.")

# Plotting
plt.figure(figsize=(8, 6))
plt.contourf(np.linspace(0, lx, nx), np.linspace(0, ly, ny), T, cmap='hot')
plt.colorbar(label='$T$ (°C)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Temperature distribution')
plt.show()