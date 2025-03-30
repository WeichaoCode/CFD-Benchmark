import numpy as np

# Domain parameters
Lx, Ly = 5.0, 4.0
dx, dy = 0.05, 0.05
nx, ny = 101, 81

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply Dirichlet boundary conditions
T[:, 0] = 10.0      # Left boundary
T[:, -1] = 40.0     # Right boundary
T[0, :] = 20.0      # Bottom boundary
T[-1, :] = 0.0      # Top boundary

# Jacobi iteration parameters
tolerance = 1e-6
max_iterations = 10000
iteration = 0
error = 1.0

# Iterative solver
while error > tolerance and iteration < max_iterations:
    T_new = T.copy()
    T_new[1:-1, 1:-1] = 0.25 * (T[1:-1, :-2] + T[1:-1, 2:] +
                                T[:-2, 1:-1] + T[2:, 1:-1])
    error = np.max(np.abs(T_new - T))
    T = T_new
    iteration += 1

# Save the final temperature field
np.save('T.npy', T)