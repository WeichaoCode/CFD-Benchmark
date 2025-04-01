import numpy as np

# Domain parameters
width = 5.0
height = 4.0
dx = 0.05
dy = 0.05
nx = 101
ny = 81

# Boundary conditions
T_left = 10.0
T_right = 40.0
T_top = 0.0
T_bottom = 20.0

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply Dirichlet boundary conditions
T[:, 0] = T_left      # Left boundary (x=0)
T[:, -1] = T_right    # Right boundary (x=5)
T[0, :] = T_bottom    # Bottom boundary (y=0)
T[-1, :] = T_top      # Top boundary (y=4)

# SOR parameters
omega = 1.5
tolerance = 1e-6
max_iterations = 10000
error = 1.0
iterations = 0

# Successive Over-Relaxation (SOR) loop
while error > tolerance and iterations < max_iterations:
    error = 0.0
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T_old = T[j, i]
            T_new = ((T[j, i+1] + T[j, i-1]) * dy**2 + (T[j+1, i] + T[j-1, i]) * dx**2) / (2 * (dx**2 + dy**2))
            T[j, i] = (1 - omega) * T_old + omega * T_new
            error = max(error, abs(T[j, i] - T_old))
    iterations += 1

# Save the final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/T_2D_Steady_Heat_Equation_SOR.npy', T)