import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
nx = 64
nz = 64
dt = 0.01
nt = 2000
nu = 1 / 5e4
D = nu / 1

# Domain
x = np.linspace(0, 1, nx)
z = np.linspace(-1, 1, nz)
X, Z = np.meshgrid(x, z)

# Initial conditions
u = 0.5 * (1 + np.tanh((Z - 0.5) / 0.1) - np.tanh((Z + 0.5) / 0.1))
w = 0.01 * np.sin(2 * np.pi * (Z + 0.5))
s = u

# Boundary conditions (periodic)
bc_u = lambda u: u
bc_w = lambda w: w
bc_s = lambda s: s

# Time stepping loop
for n in range(nt):
    # Calculate spatial derivatives
    du_dx = np.diff(u, axis=1) / (x[1] - x[0])
    dw_dx = np.diff(w, axis=1) / (x[1] - x[0])
    du_dz = np.diff(u, axis=0) / (z[1] - z[0])
    dw_dz = np.diff(w, axis=0) / (z[1] - z[0])

    # Update velocity fields
    u_new = u - dt * (u * du_dx + w * du_dz) + dt * (np.diff(np.gradient(np.gradient(u, axis=0), axis=0), axis=0) / (z[1] - z[0])**2 + np.diff(np.gradient(np.gradient(u, axis=1), axis=1), axis=1) / (x[1] - x[0])**2)
    w_new = w - dt * (u * dw_dx + w * dw_dz) + dt * (np.diff(np.gradient(np.gradient(w, axis=0), axis=0), axis=0) / (z[1] - z[0])**2 + np.diff(np.gradient(np.gradient(w, axis=1), axis=1), axis=1) / (x[1] - x[0])**2)

    # Update tracer field
    s_new = s - dt * (u * np.gradient(s, axis=1) + w * np.gradient(s, axis=0)) + dt * (np.diff(np.gradient(np.gradient(s, axis=0), axis=0), axis=0) / (z[1] - z[0])**2 + np.diff(np.gradient(np.gradient(s, axis=1), axis=1), axis=1) / (x[1] - x[0])**2)

    # Apply boundary conditions
    u = bc_u(u_new)
    w = bc_w(w_new)
    s = bc_s(s_new)

    u = u_new
    w = w_new
    s = s_new

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)