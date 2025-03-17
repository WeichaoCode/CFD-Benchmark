import numpy as np
import matplotlib.pyplot as plt

# Constants
nx, ny = 50, 50
Lx, Ly = 2.0, 1.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Initialize solutions
p = np.zeros((ny, nx))
pn = np.empty((ny, nx))
b = np.zeros((ny, nx))

# Set the source term
b[int(ny / 4), int(nx / 4)] = 100
b[int(3 * ny / 4), int(3 * nx / 4)] = -100

# Iterative solver
def laplace_2d(p, b, dx, dy, l1norm_target):
    l1norm = 1
    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2 -
                          b[1:-1, 1:-1] * dx**2 * dy**2) /
                         (2 * (dx**2 + dy**2)))
        
        # Boundary conditions
        p[0, :] = 0
        p[-1, :] = 0
        p[:, 0] = 0
        p[:, -1] = 0

        l1norm = np.sum(np.abs(p[:]) - np.abs(pn[:]))

    return p

# Call function
p = laplace_2d(p, b, dx, dy, 1e-5)

# Plotting
plt.figure(figsize=(10, 8))
plt.contourf(p)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Poisson Equation Solver')
plt.show()

# Save result
np.save('pressure_field.npy', p)