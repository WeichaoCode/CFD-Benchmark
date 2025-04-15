import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 5.0, 4.0            # Domain size
dx, dy = 0.05, 0.05          # Grid spacing
nx, ny = int(Lx/dx) + 1, int(Ly/dy) + 1  # Number of grid points

# Stability parameter
beta = dx / dy
beta_sq = beta**2

# Initialize temperature grid
T = np.zeros((ny, nx))

# Apply boundary conditions
T[:, 0] = 10.0    # Left boundary (AB)
T[:, -1] = 40.0    # Right boundary (EF)
T[0, :] = 20.0     # Bottom boundary (G)
T[-1, :] = 0.0     # Top boundary (CD)

# Convergence parameters
tolerance = 1e-6
max_iterations = 10000
iteration = 0
residual = tolerance + 1

# Gauss-Seidel Iterative Solver
while residual > tolerance and iteration < max_iterations:
    residual = 0.0
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            T_new = (T[i+1, j] + T[i-1, j] + beta_sq * (T[i, j+1] + T[i, j-1])) / (2 * (1 + beta_sq))
            delta = abs(T_new - T[i, j])
            if delta > residual:
                residual = delta
            T[i, j] = T_new
    iteration += 1
    if iteration % 100 == 0 or iteration == 1:
        print(f"Iteration {iteration}, Residual: {residual:.2e}")

print(f"Converged after {iteration} iterations with residual {residual:.2e}")

# Create coordinate arrays for plotting
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Plot contour of temperature distribution
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, T, 50, cmap='hot')
plt.colorbar(contour, label='Temperature (°C)')
plt.title('Steady-State Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Save the temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/T_2D_Steady_Heat_Equation_Gauss.npy', T)