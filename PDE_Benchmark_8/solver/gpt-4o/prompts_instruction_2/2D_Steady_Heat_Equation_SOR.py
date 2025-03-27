import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 5.0, 4.0  # Domain size
dx, dy = 0.05, 0.05  # Grid spacing
nx, ny = int(Lx/dx) + 1, int(Ly/dy) + 1  # Number of grid points
omega = 1.5  # Relaxation factor for SOR
tolerance = 1e-5  # Convergence tolerance
max_iterations = 10000  # Maximum number of iterations

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply Dirichlet boundary conditions
T[:, 0] = 10.0  # Left boundary
T[:, -1] = 40.0  # Right boundary
T[0, :] = 0.0  # Top boundary
T[-1, :] = 20.0  # Bottom boundary

# SOR iteration
for iteration in range(max_iterations):
    T_old = T.copy()
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T[j, i] = (1-omega) * T[j, i] + omega * 0.25 * (
                T[j+1, i] + T[j-1, i] + T[j, i+1] + T[j, i-1]
            )
    
    # Compute the maximum residual
    residual = np.max(np.abs(T - T_old))
    
    # Check for convergence
    if residual < tolerance:
        print(f"Converged after {iteration+1} iterations.")
        break
else:
    print("Did not converge within the maximum number of iterations.")

# Save the final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/T_2D_Steady_Heat_Equation_SOR.npy', T)

# Plot the temperature distribution
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.contourf(X, Y, T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()