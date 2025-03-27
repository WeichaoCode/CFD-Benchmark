import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 2.0, 1.0
nx, ny = 50, 50
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
tolerance = 1e-4

# Initialize p and b
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Define source term b
b[int(ny / 4), int(nx / 4)] = 100
b[int(3 * ny / 4), int(3 * nx / 4)] = -100

# Iterative solver
def solve_poisson(p, b, dx, dy, tolerance):
    pn = np.empty_like(p)
    diff = tolerance + 1  # Initialize diff to be larger than tolerance
    while diff > tolerance:
        pn[:] = p[:]
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]) +
                          dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1]) -
                          b[1:-1, 1:-1] * dx**2 * dy**2) /
                         (2 * (dx**2 + dy**2)))

        # Apply Dirichlet boundary conditions
        p[:, 0] = 0
        p[:, -1] = 0
        p[0, :] = 0
        p[-1, :] = 0

        # Calculate the difference for convergence
        diff = np.linalg.norm(p - pn, ord=2)

    return p

# Solve the Poisson equation
p = solve_poisson(p, b, dx, dy, tolerance)

# Save the final pressure field to a .npy file
np.save('pressure_field.npy', p)

# Visualization
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.contourf(X, Y, p, 20, cmap='viridis')
plt.colorbar(label='Pressure')
plt.title('Pressure Field')
plt.xlabel('x')
plt.ylabel('y')
plt.show()