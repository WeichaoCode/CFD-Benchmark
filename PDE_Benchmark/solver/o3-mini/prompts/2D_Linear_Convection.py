#!/usr/bin/env python3
import numpy as np

# Parameters
c = 1.0                     # Convection speed
x0, x1 = 0.0, 2.0           # x-domain limits
y0, y1 = 0.0, 2.0           # y-domain limits
t0, tf = 0.0, 0.50          # time domain start and end

# Grid settings
Nx = 101                    # Number of grid points in x
Ny = 101                    # Number of grid points in y
x = np.linspace(x0, x1, Nx)
y = np.linspace(y0, y1, Ny)
dx = (x1 - x0) / (Nx - 1)
dy = (y1 - y0) / (Ny - 1)

# Time stepping settings based on CFL condition
CFL = 0.5
dt = CFL * min(dx, dy) / c

# Initialize solution array (2D problem)
u = np.ones((Ny, Nx))

# Set initial condition: u = 2 in [0.5, 1] in both x and y; u = 1 elsewhere
# Note: using meshgrid with indexing='ij' so that u[j, i] corresponds to y,x
X, Y = np.meshgrid(x, y)
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2.0

# Apply Dirichlet boundary conditions (u=1 on boundaries)
u[0, :] = 1.0          # y = y0
u[-1, :] = 1.0         # y = y1
u[:, 0] = 1.0          # x = x0
u[:, -1] = 1.0         # x = x1

t = t0
while t < tf:
    # Adjust dt for final step if necessary
    if t + dt > tf:
        dt = tf - t

    # Copy current state for update
    u_new = u.copy()
    
    # Update interior points using upwind scheme:
    # u_t + c u_x + c u_y = 0  => u_new = u - c*dt/dx*(u - u_left) - c*dt/dy*(u - u_down)
    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] \
        - c * dt/dx * (u[1:-1, 1:-1] - u[1:-1, 0:-2]) \
        - c * dt/dy * (u[1:-1, 1:-1] - u[0:-2, 1:-1])
    
    # Reapply Dirichlet BCs on boundaries (u = 1)
    u_new[0, :] = 1.0
    u_new[-1, :] = 1.0
    u_new[:, 0] = 1.0
    u_new[:, -1] = 1.0

    # Update solution and time
    u = u_new
    t += dt

# Save final solution as a 2D NumPy array in 'u.npy'
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Linear_Convection.npy', u)