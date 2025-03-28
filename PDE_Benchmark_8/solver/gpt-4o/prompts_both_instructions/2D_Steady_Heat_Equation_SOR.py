import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
Lx, Ly = 5.0, 4.0
dx, dy = 0.05, 0.05
nx, ny = int(Lx/dx) + 1, int(Ly/dy) + 1

# SOR parameters
omega = 1.5
beta = dx / dy
tolerance = 1e-4

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply Dirichlet boundary conditions
T[:, 0] = 10.0  # Left boundary
T[:, -1] = 40.0  # Right boundary
T[0, :] = 20.0  # Bottom boundary
T[-1, :] = 0.0  # Top boundary

# Successive Over-Relaxation (SOR) method
converged = False
iteration = 0

while not converged:
    T_old = T.copy()
    max_change = 0.0
    
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T_new = omega * (T_old[j, i+1] + T[j, i-1] + beta**2 * (T_old[j+1, i] + T[j-1, i])) / (2 * (1 + beta**2)) + (1 - omega) * T_old[j, i]
            max_change = max(max_change, abs(T_new - T[j, i]))
            T[j, i] = T_new
    
    # Apply Dirichlet boundary conditions
    T[:, 0] = 10.0  # Left boundary
    T[:, -1] = 40.0  # Right boundary
    T[0, :] = 20.0  # Bottom boundary
    T[-1, :] = 0.0  # Top boundary
    
    # Check for convergence
    if max_change < tolerance:
        converged = True
    iteration += 1

# Save the final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/T_2D_Steady_Heat_Equation_SOR.npy', T)

# Plot the final temperature distribution
plt.figure(figsize=(8, 6))
plt.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()