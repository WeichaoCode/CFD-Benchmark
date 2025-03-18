import numpy as np
import matplotlib.pyplot as plt

# Define the computational domain
Lx = 2.0
Ly = 2.0
nx = 151
ny = 151
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Define the time-stepping parameters
nt = 300
sigma = 0.2
dt = sigma * min(dx, dy) / 2.0

# Initialize the velocity field
u = np.ones((ny, nx))
v = np.ones((ny, nx))
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    u[1:, 1:] = (un[1:, 1:] - un[1:, 1:] * dt / dx * (un[1:, 1:] - un[1:, :-1]) -
                 vn[1:, 1:] * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    v[1:, 1:] = (vn[1:, 1:] - un[1:, 1:] * dt / dx * (vn[1:, 1:] - vn[1:, :-1]) -
                 vn[1:, 1:] * dt / dy * (vn[1:, 1:] - vn[:-1, 1:]))
    
    # Apply the boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Save the final velocity field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/u_2D_Inviscid_Burgers_FOU.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/v_2D_Inviscid_Burgers_FOU.npy', v)

# Visualize the velocity field
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])
plt.show()