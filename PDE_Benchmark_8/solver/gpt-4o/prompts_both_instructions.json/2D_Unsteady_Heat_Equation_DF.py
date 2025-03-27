import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.01  # Thermal diffusivity
Q0 = 200.0  # Source term coefficient
sigma = 0.1  # Source term spread
nx, ny = 41, 41  # Grid resolution
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_max = 3.0  # Maximum time
r = 0.25  # Stability parameter

# Derived parameters
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
dt = r * dx**2 / alpha
nt = int(t_max / dt) + 1  # Number of time steps
beta2 = (dx / dy)**2

# Grid
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
T = np.zeros((nx, ny))
T_prev = np.zeros((nx, ny))

# Source term
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Time-stepping loop
for n in range(1, nt):
    T_new = np.zeros((nx, ny))
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            T_new[i, j] = (
                2 * r * (T[i+1, j] + T[i-1, j]) +
                2 * beta2 * r * (T[i, j+1] + T[i, j-1]) +
                T_prev[i, j] + 2 * dt * q[i, j]
            ) / (1 + 2 * r + 2 * beta2 * r)
    
    # Apply Dirichlet boundary conditions
    T_new[0, :] = 0
    T_new[-1, :] = 0
    T_new[:, 0] = 0
    T_new[:, -1] = 0
    
    # Update time steps
    T_prev = T.copy()
    T = T_new.copy()

# Save the final temperature field
np.save('final_temperature.npy', T)

# Visualization
plt.contourf(X, Y, T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution at t = {:.2f} s'.format(t_max))
plt.xlabel('x')
plt.ylabel('y')
plt.show()