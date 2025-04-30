import numpy as np
from numpy import array, zeros, meshgrid
import matplotlib.pyplot as plt

# Parameters
nx = 50
ny = 50
nt = 100
rho = 1.0
nu = 0.1
dt = 0.001
dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)

# Initialize arrays
u = zeros((ny, nx))
v = zeros((ny, nx))
p = zeros((ny, nx))
u_old = zeros((ny, nx))
v_old = zeros((ny, nx))

# Boundary conditions
u[:, 0] = 0
u[:, -1] = 0
v[:, 0] = 0
v[:, -1] = 0
u[0, :] = 0
u[-1, :] = 1
v[0, :] = 0
v[-1, :] = 0

# Time loop
for n in range(nt):
    # Calculate intermediate velocity
    u_old = u.copy()
    v_old = v.copy()
    u[1:-1, 1:-1] = u_old[1:-1, 1:-1] - dt/dx * (u_old[1:-1, 1:-1] * (u_old[1:-1, 1:-1] - u_old[1:-1, :-2]) ) - dt/dy * (v_old[1:-1, 1:-1] * (u_old[1:-1, 1:-1] - u_old[:-2, 1:-1])) + dt/rho * (p[1:-1, :-2] - p[1:-1, 1:-1]) + dt*nu * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])/dx**2 + dt*nu * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])/dy**2
    v[1:-1, 1:-1] = v_old[1:-1, 1:-1] - dt/dx * (u_old[1:-1, 1:-1] * (v_old[1:-1, 1:-1] - v_old[1:-1, :-2]) ) - dt/dy * (v_old[1:-1, 1:-1] * (v_old[1:-1, 1:-1] - v_old[:-2, 1:-1])) + dt/rho * (p[:-2, 1:-1] - p[1:-1, 1:-1]) + dt*nu * (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2])/dx**2 + dt*nu * (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1])/dy**2

    # Calculate pressure using Poisson equation
    p[1:-1, 1:-1] = (p[1:-1, 1:-1] + dt * rho / (dx**2 + dy**2) * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) + dt * rho / (dx**2 + dy**2) * (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]))

    # Apply boundary conditions to pressure
    p[:, 0] = 0
    p[:, -1] = 0
    p[0, :] = 0
    p[-1, :] = 0

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/u_Lid_Driven_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/v_Lid_Driven_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/p_Lid_Driven_Cavity.npy', p)