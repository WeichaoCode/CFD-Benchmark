import numpy as np

# Domain parameters
width = 5.0
height = 4.0
Nx = 50
Ny = 40
dx = width / (Nx - 1)
dy = height / (Ny - 1)

# Initialize temperature field
T = np.zeros((Ny, Nx))

# Boundary conditions
T[:, 0] = 10.0      # Left boundary (x=0)
T[:, -1] = 40.0     # Right boundary (x=5)
T[0, :] = 20.0      # Bottom boundary (y=0)
T[-1, :] = 0.0      # Top boundary (y=4)

# Iterative parameters
tolerance = 1e-4
max_iterations = 10000
error = 1.0
iteration = 0

# Iterative solver (Gauss-Seidel)
while error > tolerance and iteration < max_iterations:
    error = 0.0
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            T_old = T[i, j]
            T[i, j] = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1])
            error = max(error, abs(T[i, j] - T_old))
    iteration += 1

# Save the final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/T_2D_Steady_Heat_Equation.npy', T)