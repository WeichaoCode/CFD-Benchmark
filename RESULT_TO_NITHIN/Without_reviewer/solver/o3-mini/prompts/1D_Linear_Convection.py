#!/usr/bin/env python3
import numpy as np

# Parameters
c = 1.0
epsilon = 5e-4  # set to 0.0 for undamped, 5e-4 for damped
x_start = -5.0
x_end = 5.0
t_final = 10.0

# Discretization in space
Nx = 501
x = np.linspace(x_start, x_end, Nx)
dx = x[1] - x[0]

# Time step based on CFL conditions:
# Convection constraint: dt <= dx/c
# Diffusion constraint: dt <= dx^2/(2*epsilon) if epsilon > 0
if epsilon > 0:
    dt_c = dx / c
    dt_d = dx**2 / (2 * epsilon)
    dt = 0.4 * min(dt_c, dt_d)
else:
    dt = 0.4 * (dx / c)

# Adjust dt so that we reach t_final exactly
nt = int(t_final / dt)
dt = t_final / nt

# Initial condition
u = np.exp(-x**2)

# Time integration using forward Euler
for n in range(nt):
    # Periodic boundary conditions are enforced naturally via np.roll.
    # Upwind difference for the convection term and central difference for diffusion.
    u_x = (u - np.roll(u, 1)) / dx
    u_xx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2

    u = u - c * dt * u_x + epsilon * dt * u_xx

# Save the final solution as a 1D numpy array in "u.npy"
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_1D_Linear_Convection.npy', u)