import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Lx, Ly = 1.0, 1.0  # lengths in x and y directions
nx, ny = 50, 50  # number of points in x and y directions
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # grid spacings

# Initialize the solution
p = np.zeros((ny, nx))

# Apply boundary conditions
p[:, 0] = 0  # p = 0 at x = 0
p[:, -1] = np.linspace(0, Ly, ny)  # p = y at x = Lx
# dp/dy = 0 at y = 0, y = Ly handled automatically by using p[i-1] and p[i+1]

# Iterative solver
iter_max = 5000  # maximum number of iterations
tol = 1e-6  # convergence criterion
p_prev = p.copy()
for _ in range(iter_max):
    p[1:-1, 1:-1] = 0.25 * (p_prev[:-2, 1:-1] + p_prev[2:, 1:-1] +
                            p_prev[1:-1, :-2] + p_prev[1:-1, 2:])
    
    if np.abs(p - p_prev).max() < tol:
        break
    p_prev = p.copy()

# Create contour plot
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

plt.figure(figsize=(8, 6))
plt.contourf(x, y, p, levels=np.linspace(np.min(p), np.max(p), num=40), cmap="viridis")
plt.colorbar(label="p(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Steady-state distribution of p(x, y)")
plt.grid(True)
plt.show()