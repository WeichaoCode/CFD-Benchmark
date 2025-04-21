#!/usr/bin/env python3
import numpy as np

# Domain and grid parameters
nx = 81
ny = 81
x_start, x_end = 0.0, 2.0
y_start, y_end = 0.0, 2.0
dx = (x_end - x_start) / (nx - 1)
dy = (y_end - y_start) / (ny - 1)

x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Time parameters
t_final = 0.32
dt = 0.001  # Reduced time step for stability
nt = int(t_final / dt)

# Initialize velocity fields
u = np.ones((nx, ny))
v = np.ones((nx, ny))

# Set initial condition: u=v=2 for 0.5 <= x,y <= 1
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2.0
v[mask] = 2.0

# Time-stepping loop (explicit upwind scheme)
for n in range(nt):
    u_n = u.copy()
    v_n = v.copy()
    
    # Update interior points using first-order upwind differences
    # Since the velocity is positive in both directions, use backward differences.
    u[1:, 1:] = (u_n[1:, 1:] -
                  dt * ( u_n[1:, 1:] * (u_n[1:, 1:] - u_n[:-1, 1:]) / dx +
                         v_n[1:, 1:] * (u_n[1:, 1:] - u_n[1:, :-1]) / dy ))
    
    v[1:, 1:] = (v_n[1:, 1:] -
                  dt * ( u_n[1:, 1:] * (v_n[1:, 1:] - v_n[:-1, 1:]) / dx +
                         v_n[1:, 1:] * (v_n[1:, 1:] - v_n[1:, :-1]) / dy ))
    
    # Apply Dirichlet boundary conditions: u = v = 1 on all boundaries
    u[0, :] = 1.0
    u[-1, :] = 1.0
    u[:, 0] = 1.0
    u[:, -1] = 1.0

    v[0, :] = 1.0
    v[-1, :] = 1.0
    v[:, 0] = 1.0
    v[:, -1] = 1.0

# Save final solution arrays as .npy files
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Convection.npy', v)