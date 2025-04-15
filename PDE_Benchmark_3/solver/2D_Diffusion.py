import numpy as np
from math import pi, exp, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# parameters
Nx = 40
Ny = 40
Nt = 500
Lx = 1.0
Ly = 1.0
T = 0.1
alpha = 0.01
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = T / (Nt - 1)

# Intialize the computational grid.
X = np.linspace(0, Lx, Nx)
Y = np.linspace(0, Ly, Ny)
T = np.linspace(0, T, Nt)
u = np.zeros((Nt, Nx, Ny))
f = np.zeros((Nt, Nx, Ny))

# Manufactured solution.
usol = lambda x, y, t: exp(-t) * sin(pi*x) * sin(pi*y)

# Initialize boundary and initial conditions.

for i in range(Nx):
    for j in range(Ny):
        u[0, i, j] = usol(X[i], Y[j], T[0]) 

for n in range(Nt):
    u[n, :, 0] = usol(X[0], Y[0], T[n]) 
    u[n, :, Ny-1] = usol(X[0], Y[Ny-1], T[n]) 
    u[n, 0, :] = usol(X[0], Y[0], T[n]) 
    u[n, Nx-1, :] = usol(X[Nx-1], Y[0], T[n])

# Compute source term.
for n in range(Nt):
    for i in range(Nx):
        for j in range(Ny):
            f[n, i, j] = (
              -usol(X[i], Y[j], T[n])
              + alpha * (2.0 * pi**2 * exp(-T[n]) * sin(pi*X[i]) * sin(pi*Y[j]))
            )

# Time stepping.
for n in range(Nt-1):
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u[n+1, i, j] = (
              u[n, i, j] + dt * (
                alpha * (
                  (u[n, i+1, j] - 2.0*u[n, i, j] + u[n, i-1, j]) / dx**2 
                  + (u[n, i, j+1] - 2.0*u[n, i, j] + u[n, i, j-1]) / dy**2
                ) 
                + f[n, i, j]
              )
            )

# Plot the solution.
Xgrid, Ygrid = np.meshgrid(X, Y)

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(Xgrid, Ygrid, u[-1, :,:], rstride=2, cstride=2, \
                       cmap='viridis', linewidth=0, antialiased=False)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$u$')
plt.show()