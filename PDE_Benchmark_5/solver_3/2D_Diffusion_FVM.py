import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Definitions

h = 0.1            # duct size, m
nx = 50            # number of nodes in x direction
ny = 50            # number of nodes in y direction
dPdz = -3.2        # pressure gradient, Pa/m
mu = 1e-3          # viscosity, Pa.s

# Initialization

dx = h/(nx-1)
dy = h/(ny-1)
rho = 1            # density, kg/m^3
w = np.zeros((ny, nx))  # velocity fielf

A = np.zeros((ny*nx, ny*nx))  # coefficients matrix
B = -dPdz*np.ones(ny*nx)  # RHS vector

# Discretization

for y in range(ny):
    for x in range(nx):
        P = y*nx + x
        n = P - nx
        s = P + nx
        e = P + 1
        w = P - 1
        
        if x == 0 or x == nx-1 or y == 0 or y == ny -1:  # boundaries
            A[P, P] = 1
            B[P] = 0
        else:
            A[P, n] = mu/dy**2
            A[P, s] = mu/dy**2
            A[P, e] = mu/dx**2
            A[P, w] = mu/dx**2
            A[P, P] = -(A[P, n] + A[P, s] + A[P, e] + A[P, w])

# Solve

w_flat = solve(A, B)

# Formatting

w = w_flat.reshape(ny, nx)

# Visualization

X = np.linspace(0, h, nx)
Y = np.linspace(0, h, ny)

plt.contourf(X, Y, w)
plt.title("w-velocity distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()

plt.show()