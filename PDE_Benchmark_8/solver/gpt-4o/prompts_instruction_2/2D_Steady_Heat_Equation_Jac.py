import numpy as np
import matplotlib.pyplot as plt

# Define the domain and grid
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

# Jacobi iteration parameters
tolerance = 1e-5
max_iterations = 10000
iteration = 0
residual = np.inf

# Jacobi iteration
while residual > tolerance and iteration < max_iterations:
    T_new = T.copy()
    
    # Update the temperature field using the Jacobi method
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T_new[j, i] = 0.25 * (T[j+1, i] + T[j-1, i] + T[j, i+1] + T[j, i-1])
    
    # Compute the residual
    residual = np.max(np.abs(T_new - T))
    
    # Update the temperature field
    T = T_new
    
    # Increment the iteration counter
    iteration += 1

# Save the final temperature field to a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/T_2D_Steady_Heat_Equation_Jac.npy', T)

# Plot the temperature distribution
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.contourf(X, Y, T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()