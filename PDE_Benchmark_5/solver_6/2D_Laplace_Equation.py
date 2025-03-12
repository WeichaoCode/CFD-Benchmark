import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
Lx, Ly = 1.0, 1.0
nx, ny = 51, 51
dx, dy = Lx / (nx - 1), Ly / (ny - 1)

# Step 2: Discretize the domain
x = np.linspace(0, Lx, num=nx)
y = np.linspace(0, Ly, num=ny)
p = np.zeros((ny, nx))

# Step 3: Apply boundary conditions
p[:, 0] = 0  # p = 0 @ x = 0
p[:, -1] = y  # p = y @ x = Lx
p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = Ly

# Step 4: Iterative solver
def laplace_2d(p, y, dx, dy, l1_target):
    l1_norm = 1
    pn = np.empty_like(p)

    while l1_norm > l1_target:
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
                         dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) /
                        (2 * (dx**2 + dy**2)))

        p[:, 0] = 0  # p = 0 @ x = 0
        p[:, -1] = y  # p = y @ x = Lx
        p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = Ly

        l1_norm = (np.sum(np.abs(p[:]) - np.abs(pn[:])) /
                   np.sum(np.abs(pn[:])))

    return p

# Step 5: Visualize the solution
p = laplace_2d(p, y, dx, dy, 1e-4)

plt.figure(figsize=(8, 5))
plt.contourf(x,y,p)
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()