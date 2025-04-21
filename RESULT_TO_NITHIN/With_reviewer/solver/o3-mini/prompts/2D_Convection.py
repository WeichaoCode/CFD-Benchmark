#!/usr/bin/env python3
import numpy as np

# Domain parameters
nx = 101
ny = 101
x_start, x_end = 0.0, 2.0
y_start, y_end = 0.0, 2.0

dx = (x_end - x_start) / (nx - 1)
dy = (y_end - y_start) / (ny - 1)

# Time parameters
t_end = 0.32
dt = 0.001
nt = int(t_end / dt)

# Create the grid
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initialize u and v with initial conditions
u = np.ones((nx, ny))
v = np.ones((nx, ny))

# Set initial condition for the region 0.5 <= x <= 1 and 0.5 <= y <= 1 to 2
u[np.logical_and(X >= 0.5, X <= 1) & np.logical_and(Y >= 0.5, Y <= 1)] = 2.0
v[np.logical_and(X >= 0.5, X <= 1) & np.logical_and(Y >= 0.5, Y <= 1)] = 2.0

# Time marching loop using first-order upwind finite difference
for n in range(nt):
    # Copy the current u and v fields
    un = u.copy()
    vn = v.copy()
    
    # Update interior points with upwind scheme (assuming positive velocities)
    u[1:, 1:] = (un[1:, 1:]
                - dt * un[1:, 1:] * ((un[1:, 1:] - un[:-1, 1:]) / dx)
                - dt * vn[1:, 1:] * ((un[1:, 1:] - un[1:, :-1]) / dy))
    
    v[1:, 1:] = (vn[1:, 1:]
                - dt * un[1:, 1:] * ((vn[1:, 1:] - vn[:-1, 1:]) / dx)
                - dt * vn[1:, 1:] * ((vn[1:, 1:] - vn[1:, :-1]) / dy))
    
    # Enforce Dirichlet boundary conditions (u = 1, v = 1 on all boundaries)
    u[0, :] = 1.0
    u[-1, :] = 1.0
    u[:, 0] = 1.0
    u[:, -1] = 1.0
    
    v[0, :] = 1.0
    v[-1, :] = 1.0
    v[:, 0] = 1.0
    v[:, -1] = 1.0

# Save final solution fields for u and v as .npy files
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Convection.npy', u)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Convection.npy', v)