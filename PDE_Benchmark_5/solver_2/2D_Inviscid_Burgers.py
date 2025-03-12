import numpy as np
import matplotlib.pyplot as plt

# 1. Define parameters 
Lx, Ly, T = 1.0, 1.0, 1.0  # physical dimensions of the problem
nx, ny, nt = 100, 100, 100  # number of grid points
dx, dy, dt = Lx/(nx-1), Ly/(ny-1), T/nt  # grid spacings

# Ensuring stability via the CFL condition
cfl = min(dx, dy)**2 / max(dx, dy)
assert dt <= cfl, f'dt is greater than the cfl: {cfl}'

# 2. Discretize space and time
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
u = np.zeros((ny, nx))  # velocity field u
v = np.zeros((ny, nx))  # velocity field v

# 3. Set up the initial velocity fields.
u[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2
v[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2

# 4. Iterate using the finite difference scheme
for n in range(nt + 1): 
    un = u.copy()
    vn = v.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                     (un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])) -
                     vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]))
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     (un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])) -
                     vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]))

# 5. Visualize the velocity field using quiver plots.
X, Y = np.meshgrid(x, y)
plt.quiver(X, Y, u, v) 
plt.show()