import numpy as np
import matplotlib.pyplot as plt

# Define the domain and grid parameters
Lx, Ly = 5.0, 4.0
dx, dy = 0.05, 0.05
nx, ny = int(Lx/dx) + 1, int(Ly/dy) + 1

# Initialize the temperature field
T = np.zeros((ny, nx))

# Apply Dirichlet boundary conditions
T[:, 0] = 10.0  # Left boundary
T[:, -1] = 40.0  # Right boundary
T[0, :] = 0.0  # Top boundary
T[-1, :] = 20.0  # Bottom boundary

# Gauss-Seidel iteration parameters
tolerance = 1e-5
max_iterations = 10000
converged = False

# Gauss-Seidel iteration
for iteration in range(max_iterations):
    T_old = T.copy()
    
    # Update the temperature field using Gauss-Seidel method
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T[j, i] = 0.25 * (T[j+1, i] + T[j-1, i] + T[j, i+1] + T[j, i-1])
    
    # Compute the maximum residual
    residual = np.max(np.abs(T - T_old))
    
    # Check for convergence
    if residual < tolerance:
        converged = True
        print(f"Converged after {iteration+1} iterations.")
        break

if not converged:
    print("Did not converge within the maximum number of iterations.")

# Save the final temperature field to a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/T_2D_Steady_Heat_Equation_Gauss.npy', T)

# Plot the temperature distribution
plt.figure(figsize=(8, 6))
plt.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()