import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
Lx, Ly = 2.0, 1.0
nx, ny = 50, 50
dx, dy = Lx / (nx - 1), Ly / (ny - 1)

# Initialize p and b
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Define source terms
i1, j1 = int(nx / 4), int(ny / 4)
i2, j2 = int(3 * nx / 4), int(3 * ny / 4)
b[j1, i1] = 100
b[j2, i2] = -100

# Convergence parameters
tolerance = 1e-4
max_iterations = 10000
iteration = 0
residual = 1

# Iterative solver (Jacobi method)
while residual > tolerance and iteration < max_iterations:
    p_new = p.copy()
    # Update interior points
    p_new[1:-1,1:-1] = (
        (p[1:-1,2:] + p[1:-1,0:-2]) * dy**2 +
        (p[2:,1:-1] + p[0:-2,1:-1]) * dx**2 -
        b[1:-1,1:-1] * dx**2 * dy**2
    ) / (2 * (dx**2 + dy**2))
    
    # Apply Dirichlet boundary conditions
    p_new[0, :] = 0
    p_new[-1, :] = 0
    p_new[:, 0] = 0
    p_new[:, -1] = 0
    
    # Compute the residual
    residual = np.max(np.abs(p_new - p))
    p = p_new.copy()
    iteration += 1

# Save the final pressure field
np.save('p.npy', p)

# Create contour plot
X = np.linspace(0, Lx, nx)
Y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(X, Y)

plt.figure(figsize=(8, 4))
contour = plt.contourf(X, Y, p, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title('Pressure Contour')
plt.xlabel('x')
plt.ylabel('y')
plt.show()