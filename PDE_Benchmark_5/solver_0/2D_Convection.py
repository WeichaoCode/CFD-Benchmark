import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define parameters
Lx = 2
Ly = 2
T = 0.6  # total time
nx = 51
ny = 51
nt = 101
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = T / (nt - 1)

# Define meshgrid, CFL condition should be satisfied for stability
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)
CFL = min(dx, dy) / (4 * dt)
print(f"CFL = {CFL}")

# Define initial condition
u = np.ones((ny, nx))
v = np.ones((ny, nx))
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
v[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

# Finite difference method
for n in range(nt + 1): 
    un = u.copy()
    vn = v.copy()
    u[1:, 1:] = (un[1:, 1:] - 
                 un[1:, 1:] * dt / dx * (un[1:, 1:] - un[1:, :-1]) -
                 vn[1:, 1:] * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    v[1:, 1:] = (vn[1:, 1:] -
                 un[1:, 1:] * dt / dx * (vn[1:, 1:] - vn[1:, :-1]) -
                 vn[1:, 1:] * dt / dy * (vn[1:, 1:] - vn[:-1, 1:]))
    # Boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Draw the quiver plot
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
plt.show()