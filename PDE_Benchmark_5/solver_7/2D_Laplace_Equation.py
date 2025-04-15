import numpy as np
import matplotlib.pyplot as plt

# Step 1: Parameters
Lx, Ly = 1.0, 1.0
nx, ny = 101, 101
dx, dy = Lx/(nx-1), Ly/(ny-1)

# Step 2: Discretize Domain
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
p = np.zeros((ny, nx))


# Step 4: Apply Boundary Conditions
p[:, 0] = 0 # p = 0 @ x = 0
p[:, -1] = y # p = y @ x = Lx
p[0, :] = p[1, :] # dp/dy = 0 @ y = 0
p[-1,:] = p[-2,:] # dp/dy = 0 @ y = Ly

# Step 5: Solve PDE
def laplace_solution(p, y, dx, dy, target):
    error = 1e5
    pn = np.empty_like(p)
    while error > target:
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
                         dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) /
                        (2 * (dx**2 + dy**2)))
        
        p[:, 0] = 0  # p = 0 @ x = 0
        p[:, -1] = y # p = y @ x = Lx
        p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
        p[-1,:] = p[-2,:]  # dp/dy = 0 @ y = Ly
        error = (np.sum(np.abs(p[:]) - np.abs(pn[:])) /
                np.sum(np.abs(pn[:])))
     
    return p

p = laplace_solution(p, y, dx, dy, 1e-4)

# Step 6: Visualize the Solution
plt.figure(figsize=(8, 5))
plt.contourf(x, y, p, alpha=0.7, cmap='viridis')
plt.colorbar()
plt.title("Contour Plot of Laplace Equation")
plt.xlabel('x')
plt.ylabel('y')
plt.show()