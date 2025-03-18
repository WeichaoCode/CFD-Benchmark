import numpy as np
import matplotlib.pyplot as plt

# Define the domain
nx, ny = 50, 50
Lx, Ly = 2.0, 1.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Initialize the solution
p = np.zeros((ny, nx))

# Define the source term
b = np.zeros_like(p)
b[int(ny / 4), int(nx / 4)] = 100
b[int(3 * ny / 4), int(3 * nx / 4)] = -100

# Define the Poisson solver
def poisson_solver(p, b, dx, dy, l2_target):
    l2_norm = 1
    pn = np.empty_like(p)
    iterations = 0
    while l2_norm > l2_target:
        pn = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                         (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2 -
                         b[1:-1, 1:-1] * dx**2 * dy**2) / (2 * (dx**2 + dy**2))
        
        # Apply Dirichlet boundary conditions
        p[0, :] = 0
        p[-1, :] = 0
        p[:, 0] = 0
        p[:, -1] = 0
        
        l2_norm = np.sqrt(np.sum((p - pn)**2) / np.sum(pn**2))
        iterations += 1
        
    return p, iterations

# Solve the Poisson equation
l2_target = 1e-6
p, iterations = poisson_solver(p, b, dx, dy, l2_target)

# Save the final pressure field
np.save('pressure_field.npy', p)

# Visualize the pressure field
plt.figure(figsize=(8, 5))
plt.contourf(p, cmap='viridis')
plt.title('Pressure Field')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Pressure')
plt.show()