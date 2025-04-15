import numpy as np
import matplotlib.pyplot as plt

# Define the grid parameters
Lx, Ly = 2.0, 1.0
nx, ny = 31, 31
dx, dy = Lx / (nx - 1), Ly / (ny - 1)

# Initialize the solution array
p = np.zeros((ny, nx))

# Set the boundary conditions
p[:, 0] = 0  # left boundary
p[:, -1] = np.linspace(0, Ly, ny)  # right boundary

# Define the tolerance for convergence
tol = 1e-6
max_diff = 1.0
iterations = 0

# Jacobi iterative solver
while max_diff > tol:
    p_old = p.copy()
    p[1:-1, 1:-1] = ((dy**2 * (p_old[1:-1, 2:] + p_old[1:-1, :-2]) +
                      dx**2 * (p_old[2:, 1:-1] + p_old[:-2, 1:-1])) /
                     (2.0 * (dx**2 + dy**2)))

    # Neumann conditions at the top and bottom boundaries
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]

    max_diff = np.abs(p - p_old).max()
    iterations += 1

print(f"Jacobi method took {iterations} iterations to converge.")

# Save the final solution
np.save("/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/p_2D_Laplace_Equation.npy", p)

# Plot the solution
plt.figure(figsize=(8, 5))
plt.contourf(p, levels=50, cmap='viridis')
plt.colorbar()
plt.title("Solution to 2D Laplace Equation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()