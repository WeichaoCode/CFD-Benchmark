import numpy as np
from numpy import zeros, array
import matplotlib.pyplot as plt

# Parameters
nx = 41
ny = 41
nt = 1000
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = 0.001

rho = 1.0
nu = 0.1

# Initialize arrays
u = zeros((ny, nx))
v = zeros((ny, nx))
p = zeros((ny, nx))

# Boundary conditions
u[0, :] = 1
u[-1, :] = 0
u[:, 0] = 0
u[:, -1] = 0
v[0, :] = 0
v[-1, :] = 0
v[:, 0] = 0
v[:, -1] = 0
p[0, :] = 0
p[-1, :] = 0
p[:, 0] = 0
p[:, -1] = 0

# Time loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    pn = p.copy()

    # Update velocities
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u[j, i] = un[j, i] - dt * (
                (un[j, i] * (un[j, i] - un[j, i - 1])) / dx
                + (vn[j, i] * (un[j, i] - un[j - 1, i])) / dy
                - (1 / rho) * (pn[j, i] - pn[j, i - 1]) / dx
                + nu * (
                    (un[j, i + 1] - 2 * un[j, i] + un[j, i - 1]) / dx**2
                    + (un[j + 1, i] - 2 * un[j, i] + un[j - 1, i]) / dy**2
                )
            )
            v[j, i] = vn[j, i] - dt * (
                (un[j, i] * (vn[j, i] - vn[j, i - 1])) / dx
                + (vn[j, i] * (vn[j, i] - vn[j - 1, i])) / dy
                - (1 / rho) * (pn[j, i] - pn[j, i - 1]) / dy
                + nu * (
                    (vn[j, i + 1] - 2 * vn[j, i] + vn[j, i - 1]) / dx**2
                    + (vn[j + 1, i] - 2 * vn[j, i] + vn[j - 1, i]) / dy**2
                )
            )

    # Update pressure
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            p[j, i] = pn[j, i] - dt * (
                (1 / rho) * (
                    (un[j, i + 1] - un[j, i - 1]) / (2 * dx)
                    + (un[j + 1, i] - un[j - 1, i]) / (2 * dy)
                )
            )

# Save results
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/u_2D_Navier_Stokes_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/v_2D_Navier_Stokes_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/p_2D_Navier_Stokes_Cavity.npy', p)