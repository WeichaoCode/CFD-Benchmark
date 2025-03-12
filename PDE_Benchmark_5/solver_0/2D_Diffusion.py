import numpy as np
import matplotlib.pyplot as plt

# 1. Define parameters
Lx, Ly = 1.0, 1.0
nx, ny = 21, 21
nt = 100
v = 0.1
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
sigma = 0.25
dt = sigma * min(dx, dy)**2 / v

# Make sure dt satisfies the stability criterion
assert v * dt / dx**2 <= 0.5, "Error: Numerical scheme will be unstable"
assert v * dt / dy**2 <= 0.5, "Error: Numerical scheme will be unstable"

# 2. Initialize the grid and the initial condition
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
u = np.zeros((ny, nx)) 

# Initial condition
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2  

# 3. Solve the PDE
for t in range(nt): 
    un = u.copy()
    u[1:-1, 1:-1] = un[1:-1, 1:-1] + v * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) + \
                    v * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])

u[0, :], u[-1, :] = u[1, :], u[-2, :]
u[:, 0], u[:, -1] = u[:, 1], u[:, -2]

# 4. Visualization 
plt.figure(figsize=(8,5))
plt.contourf(x,y,u, cmap='viridis')
plt.colorbar()
plt.show()