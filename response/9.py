import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nx, ny = 101, 51  # Grid points
Lx, Ly = 2.0, 1.0  # Spatial domain: [0,2] × [0,1]
dx = Lx / (nx - 1)  # Spatial step in x
dy = Ly / (ny - 1)  # Spatial step in y
tol = 1e-4  # Convergence tolerance
max_iter = 5000  # Maximum number of iterations

# Grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize p array
p = np.zeros((ny, nx))  # Initial condition: p = 0 everywhere

# Apply boundary conditions
p[:, 0] = 0  # p = 0 at x = 0
p[:, -1] = y  # p = y at x = 2
# Neumann BC: ∂p/∂y = 0 at y = 0 and y = 1 (zero-gradient boundary)
p[0, :] = p[1, :]  # Copy second row to first row
p[-1, :] = p[-2, :]  # Copy second-last row to last row

# Solve using iterative Gauss-Seidel method
for it in range(max_iter):
    p_old = p.copy()

    # Update interior points using central difference scheme
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            p[i, j] = 0.5 * ((p[i + 1, j] + p[i - 1, j]) / dx ** 2 + (p[i, j + 1] + p[i, j - 1]) / dy ** 2) / (
                        1 / dx ** 2 + 1 / dy ** 2)

    # Reapply Neumann boundary conditions
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]

    # Check for convergence
    diff = np.max(np.abs(p - p_old))
    if diff < tol:
        print(f"Converged in {it} iterations.")
        break

# Plot the solution
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, p, cmap='jet', edgecolor='k')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("p(x, y)")
ax.set_title("2D Laplace Equation - Finite Difference Solution")
plt.show()
np.save("u_pred.npy", p)
np.save("v_pred.npy", p)
# print(p)