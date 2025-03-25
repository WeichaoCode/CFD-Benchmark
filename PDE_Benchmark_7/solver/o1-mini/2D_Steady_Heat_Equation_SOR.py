import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 5.0, 4.0       # Domain size
dx, dy = 0.05, 0.05     # Grid spacing
nx, ny = int(Lx/dx) + 1, int(Ly/dy) + 1  # Number of grid points
omega = 1.5             # Relaxation factor
tolerance = 1e-6        # Convergence criterion
max_iterations = 10000  # Maximum number of iterations

# Initialize temperature grid
T = np.zeros((nx, ny))

# Apply boundary conditions
T[0, :] = 10.0       # Left boundary (AB)
T[-1, :] = 40.0      # Right boundary (EF)
T[:, 0] = 20.0       # Bottom boundary (G)
T[:, -1] = 0.0       # Top boundary (CD)

# Iterative SOR method
for iteration in range(1, max_iterations + 1):
    T_old = T.copy()
    max_diff = 0.0

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            T_new = (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1]) / 4.0
            T_new = omega * T_new + (1 - omega) * T[i, j]
            diff = abs(T_new - T[i, j])
            if diff > max_diff:
                max_diff = diff
            T[i, j] = T_new

    # Check for convergence
    if max_diff < tolerance:
        print(f'Converged after {iteration} iterations with max difference {max_diff:.2e}.')
        break
    if iteration % 500 == 0:
        print(f'Iteration {iteration}: max difference = {max_diff:.2e}')

else:
    print(f'Did not converge within {max_iterations} iterations. Final max difference = {max_diff:.2e}.')

# Save the temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/T_2D_Steady_Heat_Equation_SOR.npy', T)

# Create coordinate arrays for plotting
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Plot contour of temperature distribution
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, T, 50, cmap='hot')
plt.title('Steady-State Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(contour, label='Temperature (°C)')
plt.tight_layout()
plt.show()