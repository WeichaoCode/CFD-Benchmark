import numpy as np
import matplotlib.pyplot as plt

# parameters
Lx, Ly = 2.0, 2.0
nx, ny = 101, 101
nt = 100
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
CFL = 0.1
dt = CFL * min(dx, dy)

# initialization
x = np.linspace(0.0, Lx, num=nx)
y = np.linspace(0.0, Ly, num=ny)
u = np.ones((ny, nx))
v = np.ones((ny, nx))
mask = np.where(np.logical_and(x >= 0.5, x <= 1.0))
u[mask] = 2.0
v[mask] = 2.0

# temporal integration
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Update u velocity with FD method
    u[1:, 1:] = (un[1:, 1:] -
                 (un[1:, 1:] * dt / dx * (un[1:, 1:] - un[1:, :-1])) -
                 (vn[1:, 1:] * dt / dy * (un[1:, 1:] - un[:-1, 1:])))

    # Update v velocity with FD method
    v[1:, 1:] = (vn[1:, 1:] -
                 (un[1:, 1:] * dt / dx * (vn[1:, 1:] - vn[1:, :-1])) -
                 (vn[1:, 1:] * dt / dy * (vn[1:, 1:] - vn[:-1, 1:])))

    # Applying Boundary conditions
    u[0, :] = 1.0
    u[-1, :] = 1.0
    u[:, 0] = 1.0
    u[:, -1] = 1.0

    v[0, :] = 1.0
    v[-1, :] = 1.0
    v[:, 0] = 1.0
    v[:, -1] = 1.0

# Plotting the velocity field with quiver plot
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(figsize=(7,7))
q = ax.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
plt.show()