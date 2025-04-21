#!/usr/bin/env python3
import numpy as np

# Parameters
u = 0.2                # convection velocity [m/s]
L = 2.0                # length of the domain [m]
T_final = 2.5          # final time [s]
m = 0.5                # center of the Gaussian [m]
s = 0.1                # width of the Gaussian

# Discretization
Nx = 201               # number of spatial cells
dx = L / (Nx - 1)      # spatial step size
x = np.linspace(0, L, Nx)

# Time step based on CFL condition
CFL = 0.5
dt = CFL * dx / u

# Initial condition: Gaussian profile
phi = np.exp(-((x - m) / s) ** 2)
phi[0] = 0.0           # Dirichlet BC at x=0
phi[-1] = 0.0          # Dirichlet BC at x=2.0

t = 0.0

while t < T_final:
    dt_current = min(dt, T_final - t)
    phi_new = np.copy(phi)
    # Upwind finite volume discretization for positive u:
    for i in range(1, Nx):
        phi_new[i] = phi[i] - u * dt_current / dx * (phi[i] - phi[i - 1])
    # Enforce Dirichlet boundary conditions
    phi_new[0] = 0.0
    phi_new[-1] = 0.0
    phi = phi_new
    t += dt_current

np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/phi_1D_Unsteady_Convection.npy', phi)