#!/usr/bin/env python3
import numpy as np

# Parameters
nu = 0.07
L = 2 * np.pi
T_final = 0.14 * np.pi  # final time
N = 256                # number of spatial grid points
dx = L / N
dt = 1e-4              # time step size
num_steps = int(T_final / dt)

# Spatial grid
x = np.linspace(0, L, N, endpoint=False)

# Compute the initial condition
phi = np.exp(-x**2 / (4 * nu)) + np.exp(-((x - L)**2) / (4 * nu))
# Analytical derivative of phi
dphi_dx = - (x / (2 * nu)) * np.exp(-x**2 / (4 * nu)) - ((x - L) / (2 * nu)) * np.exp(-((x - L)**2) / (4 * nu))
u = - (2 * nu / phi) * dphi_dx + 4

# Time integration using explicit Euler method
for step in range(num_steps):
    # Periodic boundary conditions with np.roll for spatial derivatives
    u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx ** 2)
    
    # Update equation: u_t = - u * u_x + nu * u_xx
    u = u + dt * (- u * u_x + nu * u_xx)

# Save the final time step solution as a 1D numpy array in u.npy
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_1D_Burgers_Equation.npy', u)