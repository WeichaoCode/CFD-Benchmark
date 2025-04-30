import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

nx = 41
ny = 41
nt = 81
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = 0.40 / nt

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
t = np.linspace(0, 0.40, nt)

u = np.ones((ny, nx))
v = np.ones((ny, nx))

u[int(ny / 2):int(3 * ny / 2), int(nx / 2):int(3 * nx / 2)] = 2
v[int(ny / 2):int(3 * ny / 2), int(nx / 2):int(3 * nx / 2)] = 2

for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Calculate the fluxes
    ux = (un[1:, :] - un[:-1, :]) / dx
    uy = (un[:, 1:] - un[:, :-1]) / dy
    vx = (vn[1:, :] - vn[:-1, :]) / dx
    vy = (vn[:, 1:] - vn[:, :-1]) / dy

    # Update u
    u[1:-1, 1:-1] = un[1:-1, 1:-1] - dt / dx * un[1:-1, 1:-1] * ux[1:-1, 1:-1] - dt / dy * un[1:-1, 1:-1] * uy[1:-1, 1:-1]

    # Update v
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] - dt / dx * vn[1:-1, 1:-1] * vx[1:-1, 1:-1] - dt / dy * vn[1:-1, 1:-1] * vy[1:-1, 1:-1]

    # Apply boundary conditions
    u[:, 0] = 1
    u[:, -1] = 1
    u[0, :] = 1
    u[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1

np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/u_2D_Inviscid_Burgers.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/v_2D_Inviscid_Burgers.npy', v)