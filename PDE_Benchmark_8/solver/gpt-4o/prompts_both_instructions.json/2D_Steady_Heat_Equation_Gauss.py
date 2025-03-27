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
T[0, :] = 20.0  # Bottom boundary
T[-1, :] = 0.0  # Top boundary

# Gauss-Seidel iteration parameters
beta = dx / dy
tolerance = 1e-4
max_iterations = 10000

# Gauss-Seidel iteration
for iteration in range(max_iterations):
    T_old = T.copy()
    
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T[j, i] = (T_old[j, i+1] + T[j, i-1] + beta**2 * (T_old[j+1, i] + T[j-1, i])) / (2 * (1 + beta**2))
    
    # Reapply Dirichlet boundary conditions
    T[:, 0] = 10.0
    T[:, -1] = 40.0
    T[0, :] = 20.0
    T[-1, :] = 0.0
    
    # Check for convergence
    residual = np.max(np.abs(T - T_old))
    if residual < tolerance:
        print(f'Converged after {iteration+1} iterations with residual {residual:.5e}')
        break
else:
    print('Did not converge within the maximum number of iterations')

# Save the final temperature field to a .npy file
np.save('temperature_field.npy', T)

# Plot the temperature distribution
plt.figure(figsize=(8, 6))
plt.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()