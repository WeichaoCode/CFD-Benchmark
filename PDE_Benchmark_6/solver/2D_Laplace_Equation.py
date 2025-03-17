import numpy as np
import matplotlib.pyplot as plt

# Set numerical parameters
nx, ny = 31, 31
x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)
dx, dy = x[1]-x[0], y[1]-y[0]

# Initialize solution array
p = np.zeros((ny, nx))
pn = np.empty_like(p)  # Temp array to hold updated solution

# Set boundary conditions
p[:, 0] = 0    # p = 0 at x = 0
p[:, -1] = y   # p = y at x = 2
p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 1

tolerance = 1e-4  # Convergence criterion
iter_diff = tolerance + 1.0  # Initial difference to start the loop

# Jacobi iteration loop
while iter_diff > tolerance:
    pn = p.copy()
    p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]) +
                      dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1])) /
                     (2 * (dx**2 + dy**2)))
                    
    # Compute difference between iterations
    iter_diff = (np.sum(np.abs(p[:]) - np.abs(pn[:])) /
                 np.sum(np.abs(pn[:])))

# Save solution to .npy file
np.save('laplace_solution.npy', p)

# Plot solution
plt.figure(figsize=(6,5))
plt.contourf(x, y, p, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='p')
plt.title('Solution of Laplace equation by finite difference method')
plt.show()