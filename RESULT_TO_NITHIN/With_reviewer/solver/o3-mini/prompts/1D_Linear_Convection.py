#!/usr/bin/env python3
import numpy as np

# Parameters
c = 1.0
epsilon = 5e-4  # set to 0 for undamped, 5e-4 for damped
x_start, x_end = -5.0, 5.0
t_final = 10.0

# Numerical discretization
Nx = 501
dx = (x_end - x_start) / (Nx - 1)
x = np.linspace(x_start, x_end, Nx)

# Time step (satisfying CFL condition for advection)
dt = 0.005
Nt = int(t_final / dt)

# Initial condition: u(x,0) = exp(-x^2)
u = np.exp(-x**2)

# Time integration (explicit Euler)
for n in range(Nt):
    # Periodic boundary: use np.roll for periodic indexing
    u_im1 = np.roll(u, 1)   # u[i-1]
    u_ip1 = np.roll(u, -1)  # u[i+1]
    
    # Spatial derivatives
    # Use backward difference for convection (upwind scheme assuming c>0)
    dudx = (u - u_im1) / dx
    # Central difference for the second derivative (diffusion term)
    d2udx2 = (u_ip1 - 2*u + u_im1) / dx**2
    
    # Update solution
    u = u + dt * (-c * dudx + epsilon * d2udx2)

# Save final solution
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/u_1D_Linear_Convection.npy', u)