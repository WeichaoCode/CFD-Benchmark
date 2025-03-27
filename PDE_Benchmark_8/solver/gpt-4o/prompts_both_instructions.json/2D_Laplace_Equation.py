import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 31, 31
dx = 2 / (nx - 1)
dy = 1 / (ny - 1)
tolerance = 1e-5

# Initialize the potential field
p = np.zeros((ny, nx))

# Boundary conditions
p[:, 0] = 0  # Left boundary
p[:, -1] = np.linspace(0, 1, ny)  # Right boundary

# Iterative solver using Jacobi method
def solve_laplace(p, dx, dy, tolerance):
    pn = np.empty_like(p)
    diff = tolerance + 1  # Initialize difference to enter the loop
    while diff > tolerance:
        pn[:] = p[:]
        # Update the potential field using the 5-point stencil
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]) +
                          dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1])) /
                         (2 * (dx**2 + dy**2)))
        
        # Neumann boundary conditions (top and bottom)
        p[0, :] = p[1, :]  # Top boundary
        p[-1, :] = p[-2, :]  # Bottom boundary
        
        # Reapply Dirichlet boundary conditions
        p[:, 0] = 0  # Left boundary
        p[:, -1] = np.linspace(0, 1, ny)  # Right boundary
        
        # Calculate the maximum difference for convergence check
        diff = np.max(np.abs(p - pn))
    
    return p

# Solve the PDE
p_final = solve_laplace(p, dx, dy, tolerance)

# Save the final solution to a .npy file
np.save('laplace_solution.npy', p_final)

# Plot the final solution
plt.figure(figsize=(8, 4))
plt.contourf(np.linspace(0, 2, nx), np.linspace(0, 1, ny), p_final, 20, cmap='viridis')
plt.colorbar(label='Potential')
plt.title('Laplace Equation Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()