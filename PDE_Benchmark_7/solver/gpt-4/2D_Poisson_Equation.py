import numpy as np
import matplotlib.pyplot as plt

# Define computational grid
nx, ny = 50, 50
Lx, Ly = 2.0, 1.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Initialize solution array
p = np.zeros((ny, nx))

# Define source term
b = np.zeros_like(p)
b[int(ny / 4), int(nx / 4)] = 100
b[int(3 * ny / 4), int(3 * nx / 4)] = -100

# Define tolerance for convergence
tol = 1e-6
maxiter = 20000

def laplace2d(p, b, dx, dy, tol, maxiter):
    iter = 0
    err = 1e10  # Initial error value

    while err > tol and iter < maxiter:
        pn = p.copy()
        
        # Update the solution at interior points
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                         (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2 -
                         b[1:-1, 1:-1] * dx**2 * dy**2) / (2 * (dx**2 + dy**2))
        
        # Apply boundary conditions
        p[:, 0] = 0    # p = 0 at x = 0
        p[:, -1] = 0   # p = 0 at x = 2
        p[0, :] = 0    # p = 0 at y = 0
        p[-1, :] = 0   # p = 0 at y = 1

        # Compute residual
        err = np.linalg.norm(p - pn, 2)
        
        iter += 1

    return p, iter

# Solve the Poisson equation
p, iter = laplace2d(p, b, dx, dy, tol, maxiter)

# Save the solution to a file
np.save('pressure_field.npy', p)

# Plot the solution
plt.figure(figsize=(8,5))
plt.contourf(p, cmap='viridis')
plt.title('2D Poisson Equation solution')
plt.xlabel('X Index')
plt.ylabel('Y Index')
plt.colorbar(label='Pressure')
plt.show()