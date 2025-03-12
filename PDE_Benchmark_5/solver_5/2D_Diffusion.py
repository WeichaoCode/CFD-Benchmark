import numpy as np
import matplotlib.pyplot as plt

# Define constants and discretization parameters
Lx, Ly = 1.0, 1.0  # box length
nx, ny = 100, 100  # number of grid points
nt = 500  # number of time steps
v = 0.1  # viscosity/diffusion coefficient
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # grid spacing
dt = (dx**2 * dy**2) / (2 * v * (dx**2 + dy**2))  # time step (stability condition)

# Initialize u and set boundary conditions
u = np.zeros((nx, ny))
u[int(0.5 / dx):int(1 / dx + 1), int(0.5 / dy):int(1 / dy + 1)] = 1

for n in range(nt):
    un = u.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                     v * dt / dx**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) +
                     v * dt / dy**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]))
    u[0, :], u[-1, :], u[:, 0], u[:, -1] = [0, 0, 0, 0]  # boundary conditions

# Plot
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(5, 5), dpi=100)
plt.pcolormesh(X, Y, u, shading='auto', cmap='hot')
plt.colorbar()
plt.show()