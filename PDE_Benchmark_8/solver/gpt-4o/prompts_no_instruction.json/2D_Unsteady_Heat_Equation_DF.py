import numpy as np

# Parameters
alpha = 0.01  # thermal diffusivity
Q0 = 200.0  # source term coefficient
sigma = 0.1  # source term spread
nx, ny = 41, 41  # grid resolution
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_max = 3.0  # maximum time
r = 0.25  # stability parameter

# Grid setup
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Time step
dt = r * dx**2 / alpha

# Initialize temperature field
T = np.zeros((ny, nx))
T_new = np.zeros((ny, nx))
T_old = np.zeros((ny, nx))

# Source term
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Time-stepping loop
t = 0.0
while t < t_max:
    # Update interior points using DuFort-Frankel method
    T_new[1:-1, 1:-1] = (
        (1 - 2 * r) * T_old[1:-1, 1:-1] +
        2 * r * (T[1:-1, 2:] + T[1:-1, :-2] + T[2:, 1:-1] + T[:-2, 1:-1]) / 2 +
        2 * dt * q[1:-1, 1:-1]
    ) / (1 + 2 * r)

    # Apply Dirichlet boundary conditions
    T_new[0, :] = 0
    T_new[-1, :] = 0
    T_new[:, 0] = 0
    T_new[:, -1] = 0

    # Update time and swap arrays
    t += dt
    T_old, T = T, T_new

# Save the final temperature field
np.save('final_temperature.npy', T)