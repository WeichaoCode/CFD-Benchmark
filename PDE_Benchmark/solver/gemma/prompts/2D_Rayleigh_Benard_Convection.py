import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

Lx = 4
Lz = 1
Nt = 50
Nx = 64
Nz = 64
dt = 0.01
Ra = 2e6
Pr = 1
nu = (Ra/Pr)**(-0.5)
kappa = (Ra*Pr)**(-0.5)

x = np.linspace(0, Lx, Nx)
z = np.linspace(0, Lz, Nz)
X, Z = np.meshgrid(x, z)

u = np.zeros((Nx, Nz))
w = np.zeros((Nx, Nz))
p = np.zeros((Nx, Nz))
b = np.zeros((Nx, Nz))

b[:, :] = Lz - Z + np.random.rand(Nx, Nz) * 0.1

def laplacian(f):
    df_xx = np.diff(np.diff(f, axis=0), axis=0)
    df_zz = np.diff(np.diff(f, axis=1), axis=1)
    return df_xx + df_zz

def advection(f, u, w):
    du_dx = np.diff(u, axis=0)
    dw_dz = np.diff(w, axis=1)
    return -np.roll(u * du_dx, 1, axis=0) - np.roll(w * dw_dz, 1, axis=1)

for n in range(Nt):
    dt_u = dt * (np.linalg.norm(u) + np.linalg.norm(w))
    dt_b = dt * (np.linalg.norm(b))
    dt = min(dt_u, dt_b)

    u_new = u + dt * (advection(u, u, w) + nu * laplacian(u))
    w_new = w + dt * (advection(w, u, w) + nu * laplacian(w))
    b_new = b + dt * (advection(b, u, w) + kappa * laplacian(b))

    u = u_new
    w = w_new
    b = b_new

np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/u_2D_Rayleigh_Benard_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/w_2D_Rayleigh_Benard_Convection.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/p_2D_Rayleigh_Benard_Convection.npy', p)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/b_2D_Rayleigh_Benard_Convection.npy', b)