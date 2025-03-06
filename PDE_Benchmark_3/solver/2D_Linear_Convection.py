import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse.linalg as sp

# Step 1: Define parameters
Lx = 1.0        # length of domain in x
Ly = 1.0        # length of domain in y
T_final = 1.0   # final time
nx = 21         # grid points in x
ny = 21         # grid points in y
nt = 251        # time steps
dt = T_final / (nt - 1)  # time step
dx = Lx / (nx - 1)       # grid resolution in x
dy = Ly / (ny - 1)       # grid resolution in y
cx = 1.0        # speed of convection in x
cy = 1.0        # speed of convection in y


# Step 2: Check CFL condition
CFL = max(cx * dt / dx, cy * dt / dy)
if CFL >= 1:
    print('CFL condition not satisfied: ', CFL, '>= 1')
    dt = 0.8 / max(cx / dx, cy / dy)
    nt = int(T_final / dt) + 1
else:
    print('CFL condition satisfied: ', CFL, '< 1')


# Step 3: Compute source term
X, Y, T = np.meshgrid(np.linspace(0, Lx, nx),
                      np.linspace(0, Ly, ny),
                      np.linspace(0, T_final, nt), indexing='ij')

U_exact = np.exp(-T) * np.sin(np.pi * X) * np.sin(np.pi * Y)

# Compute source term f from MMS
f = -np.exp(-T) * np.sin(np.pi * X) * np.sin(np.pi * Y) \
    + np.pi * np.exp(-T) * (cx * np.cos(np.pi * X) * np.sin(np.pi * Y) \
                            + cy * np.sin(np.pi * X) * np.cos(np.pi * Y))


# Step 4: Compute initial and boundary conditions from MMS
U = np.zeros((nx, ny, nt))
U[:, :, 0] = U_exact[:, :, 0]  # initial condition
U[0, :, :] = U_exact[0, :, :]  # x=0 boundary
U[-1, :, :] = U_exact[-1, :, :]  # x=Lx boundary
U[:, 0, :] = U_exact[:, 0, :]  # y=0 boundary
U[:, -1, :] = U_exact[:, -1, :]  # y=Ly boundary


# Step 5: Solve PDE using finite difference
for n in range(nt-1):
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            U[i, j, n+1] = (
                U[i, j, n] 
                - dt / dx * cx * (U[i, j, n] - U[i-1, j, n]) 
                - dt / dy * cy * (U[i, j, n] - U[i, j-1, n]) 
                + dt * f[i, j, n])


# Step 6: Compute exact solution for comparison
U_exact_final = np.exp(-T_final) * np.sin(np.pi * X[:, :, 0]) * np.sin(np.pi * Y[:, :, 0])


# Step 7: Error analysis and plot
error = np.abs(U_exact_final - U[:, :, -1])
print('Maximum error: ', error.max())

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
ax.plot_surface(X[:, :, 0], Y[:, :, 0], U[:, :, -1], cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
ax.plot_surface(X[:, :, 0], Y[:, :, 0], U_exact_final, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
ax.plot_surface(X[:, :, 0], Y[:, :, 0], error, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()