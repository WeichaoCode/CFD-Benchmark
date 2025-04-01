import numpy as np

# Domain parameters
Lx, Ly = 5.0, 4.0
nx, ny = 101, 81
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize temperature array
T = np.zeros((ny, nx))

# Apply boundary conditions
T[0, :] = 20.0  # Bottom boundary
T[-1, :] = 0.0  # Top boundary
T[:, 0] = 10.0  # Left boundary
T[:, -1] = 40.0  # Right boundary

# SOR parameters
omega = 1.8  # Relaxation parameter
max_iter = 10000
tolerance = 1e-6

# SOR iteration
for _ in range(max_iter):
    T_old = T.copy()
    
    # Update interior points using SOR
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T[j, i] = (1 - omega) * T_old[j, i] + \
                      omega * 0.25 * (T[j, i+1] + T[j, i-1] + 
                                      T[j+1, i] + T[j-1, i])
    
    # Check convergence
    if np.max(np.abs(T - T_old)) < tolerance:
        break

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/T_2D_Steady_Heat_Equation_SOR.npy', T)