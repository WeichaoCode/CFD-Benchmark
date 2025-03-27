import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 1e-3  # dynamic viscosity (Pa·s)
dPdz = -3.2  # pressure gradient (Pa/m)
h = 0.1  # domain height (m)
n_x = n_y = 80  # number of grid points
dx = dy = h / (n_x - 1)  # grid spacing

# Initialize the velocity field
w = np.zeros((n_x, n_y))

# Coefficients for the finite volume method
alpha = mu / dx**2
beta = mu / dy**2
gamma = -dPdz

# Jacobi iteration parameters
tolerance = 1e-6
max_iterations = 10000

# Iterative solver using Jacobi method
for iteration in range(max_iterations):
    w_new = np.copy(w)
    
    # Update the interior points
    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            w_new[i, j] = (alpha * (w[i+1, j] + w[i-1, j]) +
                           beta * (w[i, j+1] + w[i, j-1]) -
                           gamma) / (2 * (alpha + beta))
    
    # Check for convergence
    if np.linalg.norm(w_new - w, ord=np.inf) < tolerance:
        print(f"Converged after {iteration} iterations")
        break
    
    w = w_new

# Visualization
plt.figure(figsize=(8, 6))
plt.contourf(w, levels=50, cmap='viridis')
plt.colorbar(label='Velocity w (m/s)')
plt.title('Velocity Contour Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Save the final solution to a .npy file
np.save('velocity_field.npy', w)