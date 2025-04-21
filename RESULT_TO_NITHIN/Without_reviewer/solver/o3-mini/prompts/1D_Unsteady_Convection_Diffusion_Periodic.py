#!/usr/bin/env python3
import numpy as np

# Parameters
u = 0.2                   # convection speed (m/s)
rho = 1.0                 # density (kg/m^3)
Gamma = 0.001             # diffusion coefficient (kg/(m·s))

# Domain
x_start = 0.0
x_end = 2.0
t_start = 0.0
t_final = 2.5

# Numerical parameters
N = 101                   # number of control volumes
L = x_end - x_start
dx = L / N

# Create cell centers (1D array)
x = np.linspace(x_start, x_end - dx, N)

# Initial condition: Gaussian profile centered at m=0.5, width s=0.1
m = 0.5
s = 0.1
phi = np.exp(-((x - m)/s)**2)

# Time step based on CFL conditions: convection and diffusion criteria
CFL = 0.5
dt_conv = CFL * dx / u
dt_diff = CFL * dx**2 / (Gamma / rho)
dt = min(dt_conv, dt_diff)

# Time stepping using explicit Euler
time = t_start
while time < t_final:
    # Adjust dt for final step
    if time + dt > t_final:
        dt = t_final - time
    # Periodic BCs handled via np.roll operator
    # Upwind scheme for convection (u > 0) and central difference for diffusion
    phi_conv = (phi - np.roll(phi, 1)) / dx
    phi_diff = (np.roll(phi, -1) - 2*phi + np.roll(phi, 1)) / dx**2
    phi = phi - dt * u * phi_conv + dt * (Gamma / rho) * phi_diff
    time += dt

# Save the final solution as a 1D NumPy array in phi.npy
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/phi_1D_Unsteady_Convection_Diffusion_Periodic.npy', phi)