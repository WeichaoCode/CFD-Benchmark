import numpy as np

# Parameters
alpha = 0.01  # Thermal diffusivity
Q0 = 200.0  # Source term coefficient
sigma = 0.1  # Source term spread
nx, ny = 41, 41  # Grid resolution
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_max = 3.0  # Maximum time

# Spatial discretization
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)

# Stability condition
beta = dx / dy
r = 0.25 / (1 + beta**2)  # Ensuring stability condition
dt = r * dx**2 / alpha

# Time discretization
nt = int(t_max / dt) + 1

# Initialize temperature field
T = np.zeros((nx, ny))

# Source term
X, Y = np.meshgrid(x, y, indexing='ij')
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Time-stepping loop
for n in range(nt):
    Tn = T.copy()
    T[1:-1, 1:-1] = (
        r * (Tn[2:, 1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1]) +
        beta**2 * r * (Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) +
        Tn[1:-1, 1:-1] + dt * q[1:-1, 1:-1]
    )
    # Apply Dirichlet boundary conditions
    T[:, 0] = 0
    T[:, -1] = 0
    T[0, :] = 0
    T[-1, :] = 0

# Save the final temperature field
np.save('final_temperature.npy', T)