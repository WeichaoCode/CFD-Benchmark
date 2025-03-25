import numpy as np
import matplotlib.pyplot as plt

# Define the domain
nx, ny = 31, 31
x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)
dx, dy = x[1]-x[0], y[1]-y[0]

# Initialize the solution
p = np.zeros((ny, nx))

# Set boundary conditions
p[:, 0] = 0  # left boundary
p[:, -1] = y  # right boundary

# Define the solver
def laplace_solver(p, dx, dy, l1norm_target):
    l1norm = 1
    pn = np.empty_like(p)

    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
                         dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) /
                        (2 * (dx**2 + dy**2)))
            
        # Neumann conditions
        p[0, :] = p[1, :]  # bottom boundary
        p[-1, :] = p[-2, :]  # top boundary
        
        l1norm = (np.sum(np.abs(p[:]) - np.abs(pn[:])) /
                np.sum(np.abs(pn[:])))
     
    return p

# Solve
p = laplace_solver(p, dx, dy, 1e-4)

# Plot
plt.figure(figsize=(8,5))
plt.contourf(x,y,p,100,cmap='jet')
plt.title('2D Laplace Equation')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()

# Save the final p(x,y) in .npy format
np.save('p_solution.npy', p)