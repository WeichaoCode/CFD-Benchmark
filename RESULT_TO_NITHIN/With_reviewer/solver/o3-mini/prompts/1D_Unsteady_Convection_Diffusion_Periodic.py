#!/usr/bin/env python3
import numpy as np

# Physical parameters
u = 0.2          # m/s
rho = 1.0        # kg/m^3
Gamma = 0.001    # kg/(m·s)

# Domain parameters
x_left = 0.0
x_right = 2.0
T_final = 2.5

# Gaussian initial condition parameters
m = 0.5
s = 0.1

# Numerical parameters
Nx = 101                     # number of cells
dx = (x_right - x_left) / (Nx - 1)
# Choose time step based on CFL condition
CFL = 0.5
dt_conv = CFL * dx / u
dt_diff = CFL * dx**2 * rho / Gamma  # diffusion stability condition
dt = min(dt_conv, dt_diff)
# Ensure that T_final is an integer multiple of dt
Ntimesteps = int(np.ceil(T_final / dt))
dt = T_final / Ntimesteps

# Create spatial grid (cell centers)
x = np.linspace(x_left, x_right, Nx)

# Initial condition
phi = np.exp(-((x - m) / s)**2)

# Time-stepping loop (explicit Euler)
for n in range(Ntimesteps):
    # Compute periodic shifts
    phi_left = np.roll(phi, 1)
    phi_right = np.roll(phi, -1)
    
    # Upwind scheme for convection (u > 0, so use left value)
    convection = u * (phi - phi_left) / dx
    
    # Central difference for diffusion
    diffusion = (Gamma/(rho * dx**2)) * (phi_right - 2*phi + phi_left)
    
    # Update phi
    phi = phi - dt * convection + dt * diffusion

# Save the final solution as a 1D numpy array in "phi.npy"
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/phi_1D_Unsteady_Convection_Diffusion_Periodic.npy', phi)