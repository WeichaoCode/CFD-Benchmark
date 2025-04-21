#!/usr/bin/env python3
import numpy as np

# Parameters
u = 0.2            # m/s, convection velocity
L = 2.0            # Length of the domain in x-direction
t_final = 2.5      # Final time
m = 0.5            # Center of the Gaussian
s = 0.1            # Spread (sigma) of the Gaussian

# Numerical parameters
nx = 201            # Number of cells
dx = L / (nx - 1)   # Spatial resolution
CFL = 0.9           # CFL number for stability
dt = CFL * dx / u   # Time step computed from CFL condition

# Generate the spatial grid
x = np.linspace(0, L, nx)

# Initial condition: Gaussian concentration profile
phi = np.exp(-((x - m) / s) ** 2)

# Apply Dirichlet boundary conditions at initial time
phi[0] = 0.0
phi[-1] = 0.0

# Time marching loop: unsteady problem
t = 0.0
while t < t_final:
    # Adjust dt if the final step overshoots t_final
    if t + dt > t_final:
        dt = t_final - t
    
    # Create a copy of phi for the new time step
    phi_new = np.copy(phi)
    
    # Upwind finite volume discretization for convection (u>0)
    # Loop over interior cells only
    for i in range(1, nx):
        phi_new[i] = phi[i] - u * dt / dx * (phi[i] - phi[i-1])
    
    # Enforce Dirichlet boundary conditions
    phi_new[0] = 0.0
    phi_new[-1] = 0.0
    
    # Update phi and time
    phi = phi_new
    t += dt

# Save the final solution field as a 1D numpy array in 'phi.npy'
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/phi_1D_Unsteady_Convection.npy', phi)