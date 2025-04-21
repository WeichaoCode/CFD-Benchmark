#!/usr/bin/env python3
import numpy as np

# Parameters
gamma = 1.4
nx = 400              # number of spatial grid points
x_start = -1.0
x_end = 1.0
t_final = 0.25
cfl = 0.5

# Spatial grid
x = np.linspace(x_start, x_end, nx)
dx = (x_end - x_start) / (nx - 1)

# Allocate conservative variable array U = [rho, rho*u, rho*E] for each grid point
U = np.zeros((nx, 3))

# Set initial conditions
# Left region: x < 0
rho_L = 1.0
u_L = 0.0
p_L = 1.0
# Right region: x >= 0
rho_R = 0.125
u_R = 0.0
p_R = 0.1

for i in range(nx):
    if x[i] < 0:
        rho = rho_L
        u = u_L
        p = p_L
    else:
        rho = rho_R
        u = u_R
        p = p_R
    E = p / ((gamma - 1) * rho) + 0.5 * u**2
    U[i, 0] = rho
    U[i, 1] = rho * u
    U[i, 2] = rho * E

def compute_flux(U):
    """
    Compute the flux vector F = [rho*u, rho*u**2 + p, u*(rho*E+p)]
    for the Euler equations.
    U is an array of shape (nx,3): [rho, rho*u, rho*E]
    """
    rho = U[:, 0]
    u = U[:, 1] / rho
    E = U[:, 2] / rho
    p = (gamma - 1) * (U[:, 2] - 0.5 * U[:, 1]**2 / rho)
    
    F = np.zeros_like(U)
    F[:, 0] = rho * u
    F[:, 1] = rho * u**2 + p
    F[:, 2] = u * (rho * E + p)
    return F

def max_wave_speed(U):
    """
    Compute the maximum wave speed max(|u|+c) for each cell.
    """
    rho = U[:, 0]
    u = U[:, 1] / rho
    E = U[:, 2] / rho
    p = (gamma - 1) * (U[:, 2] - 0.5 * U[:, 1]**2 / rho)
    c = np.sqrt(gamma * p / rho)
    return np.max(np.abs(u) + c)

# Time integration using Lax-Friedrich method
t = 0.0
while t < t_final:
    # Compute time step based on CFL condition
    max_lambda = max_wave_speed(U)
    dt = cfl * dx / max_lambda
    if t + dt > t_final:
        dt = t_final - t

    F = compute_flux(U)
    U_new = np.copy(U)
    # Lax-Friedrich update for interior points
    for i in range(1, nx-1):
        U_new[i, :] = 0.5*(U[i+1, :] + U[i-1, :]) - dt/(2*dx)*(F[i+1, :] - F[i-1, :])
    
    # Apply reflective (no-flux) boundary conditions
    # Left boundary (i=0): reflect velocity
    U_new[0, 0] = U_new[1, 0]
    U_new[0, 1] = - U_new[1, 1]
    U_new[0, 2] = U_new[1, 2]
    # Right boundary (i=nx-1): reflect velocity
    U_new[-1, 0] = U_new[-2, 0]
    U_new[-1, 1] = - U_new[-2, 1]
    U_new[-1, 2] = U_new[-2, 2]
    
    U = U_new
    t += dt

# Save the variables at the final time step in .npy files (1D arrays)
density = U[:, 0]
momentum = U[:, 1]
energy = U[:, 2]

np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/density_1D_Euler_Shock_Tube.npy', density)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/momentum_1D_Euler_Shock_Tube.npy', momentum)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/energy_1D_Euler_Shock_Tube.npy', energy)