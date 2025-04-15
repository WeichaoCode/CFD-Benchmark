import numpy as np
import matplotlib.pyplot as plt

# Computational domain
Lx = Ly = 2
nx = ny = 101
nt = 100
dx = Lx/(nx - 1)
dy = Ly/(ny - 1)
dt = 0.01

# Initial conditions
u = np.ones((ny, nx))
v = np.ones((ny, nx))

u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
v[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2

# Compute and apply finite difference for each time step
for n in range(nt + 1):
    un = u.copy()
    vn = v.copy()
    
    u[1:, 1:] = (un[1:, 1:] - dt/dx*un[1:, 1:]*(un[1:, 1:] - un[1:, :-1]) -
                 dt/dy*vn[1:, 1:]*(un[1:, 1:] - un[:-1, 1:]))
    v[1:, 1:] = (vn[1:, 1:] - dt/dx*un[1:, 1:]*(vn[1:, 1:] - vn[1:, :-1]) -
                 dt/dy*vn[1:, 1:]*(vn[1:, 1:] - vn[:-1, 1:]))
    
    # Apply boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Plotting the solution using quiver plot
Y, X = np.mgrid[0:2:(nx*1j), 0:2:(ny*1j)]
plt.quiver(X, Y, u, v)
plt.title('Evolution of velocity field')
plt.show()