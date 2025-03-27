import numpy as np

# Parameters
alpha = 0.01  # Thermal diffusivity
Q0 = 200.0  # Source term coefficient
sigma = 0.1  # Source term spread
nx, ny = 41, 41  # Grid resolution
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_max = 3.0  # Maximum time
r = 0.25  # Stability parameter

# Discretization
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
dt = r * dx**2 / alpha
nt = int(t_max / dt) + 1

# Grid
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
T = np.zeros((nx, ny))
T_new = np.zeros((nx, ny))
T_old = np.zeros((nx, ny))

# Source term
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Time-stepping loop
for n in range(1, nt):
    T_old[:, :] = T[:, :]
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            T_new[i, j] = (
                2 * r * (T[i+1, j] + T[i-1, j]) +
                2 * r * (T[i, j+1] + T[i, j-1]) +
                T_old[i, j] + 2 * dt * q[i, j]
            ) / (1 + 4 * r)
    T[:, :] = T_new[:, :]

# Save the final solution
np.save('final_temperature.npy', T)