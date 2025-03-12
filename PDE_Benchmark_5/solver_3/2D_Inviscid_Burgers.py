import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Lx = 1.0
Ly = 1.0
T = 0.1
nx = 101
ny = 101
nt = 100
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = T / nt
cfl = 0.1  # CFL number

# Discretize space and time
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initial velocity field
u = np.ones([nx, ny])  # Horizontal velocity
v = np.ones([nx, ny])  # Vertical velocity

u[int(.5 / dx): int(1 / dx + 1), int(.5 / dy): int(1 / dy + 1)] = 2
v[int(.5 / dx): int(1 / dx + 1), int(.5 / dy): int(1 / dy + 1)] = 2

un = np.ones([nx, ny])
vn = np.ones([nx, ny])

# Iterate using FDM
for n in range(nt + 1):
    un = u.copy()
    vn = v.copy()
    u[1:, 1:] = (un[1:, 1:] - un[1:, 1:] * dt / dx * 
                (un[1:, 1:] - un[1:, :-1]) - vn[1:, 1:] *
                dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    v[1:, 1:] = (vn[1:, 1:] - un[1:, 1:] * dt / dx * 
                (vn[1:, 1:] - vn[1:, :-1]) - vn[1:, 1:] *
                dt / dy * (vn[1:, 1:] - vn[:-1, 1:])) 
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1
# Plot the velocity field
plt.quiver(x[::3], y[::3], u[::3, ::3], v[::3, ::3])
plt.show()