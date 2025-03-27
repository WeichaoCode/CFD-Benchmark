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

# Derived parameters
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
dt = r * dx**2 / alpha
nt = int(t_max / dt) + 1  # Number of time steps

# Grid
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
T = np.zeros((ny, nx))

# Source term
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# ADI method
for n in range(nt):
    # Intermediate step (x-direction implicit)
    T_half = np.copy(T)
    for j in range(1, ny-1):
        A = np.zeros((nx, nx))
        B = np.zeros(nx)
        for i in range(1, nx-1):
            A[i, i-1] = -0.5 * r
            A[i, i] = 1 + r
            A[i, i+1] = -0.5 * r
            B[i] = 0.5 * r * (T[i+1, j] + T[i-1, j]) + (1 - r) * T[i, j] + 0.5 * dt * q[i, j]
        A[0, 0] = A[-1, -1] = 1  # Dirichlet boundary conditions
        T_half[:, j] = np.linalg.solve(A, B)

    # Final step (y-direction implicit)
    T_new = np.copy(T_half)
    for i in range(1, nx-1):
        A = np.zeros((ny, ny))
        B = np.zeros(ny)
        for j in range(1, ny-1):
            A[j, j-1] = -0.5 * r
            A[j, j] = 1 + r
            A[j, j+1] = -0.5 * r
            B[j] = 0.5 * r * (T_half[i, j+1] + T_half[i, j-1]) + (1 - r) * T_half[i, j] + 0.5 * dt * q[i, j]
        A[0, 0] = A[-1, -1] = 1  # Dirichlet boundary conditions
        T_new[i, :] = np.linalg.solve(A, B)

    T = T_new

# Save the final solution
np.save('final_temperature.npy', T)