import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.01  # thermal diffusivity
Q0 = 200.0  # source term strength
sigma = 0.1  # source term spread
nx, ny = 41, 41  # grid resolution
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
r = 0.25  # stability factor
dt = r * dx**2 / alpha
t_max = 3.0
nt = int(t_max / dt)  # number of time steps

# Initialize temperature field
T = np.zeros((nx, ny))
T_prev = np.zeros((nx, ny))
T_next = np.zeros((nx, ny))

# Source term
X, Y = np.meshgrid(x, y, indexing='ij')
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# DuFort-Frankel method
for n in range(nt):
    # Update interior points
    T_next[1:-1, 1:-1] = (
        (1 - 2 * r) * T_prev[1:-1, 1:-1] +
        2 * r * (
            T[2:, 1:-1] + T[:-2, 1:-1] +
            T[1:-1, 2:] + T[1:-1, :-2]
        ) +
        2 * dt * q[1:-1, 1:-1]
    ) / (1 + 2 * r)

    # Apply Dirichlet boundary conditions
    T_next[0, :] = 0
    T_next[-1, :] = 0
    T_next[:, 0] = 0
    T_next[:, -1] = 0

    # Update time steps
    T_prev, T = T, T_next

# Save the final temperature field
np.save('final_temperature.npy', T)

# Visualization
plt.contourf(X, Y, T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution at t = {:.2f} s'.format(t_max))
plt.xlabel('x')
plt.ylabel('y')
plt.show()