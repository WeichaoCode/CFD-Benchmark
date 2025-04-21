#!/usr/bin/env python3
import numpy as np

# Domain parameters
nx = 101            # number of grid points in x
ny = 101            # number of grid points in y
x_start, x_end = 0.0, 2.0
y_start, y_end = 0.0, 2.0

dx = (x_end - x_start) / (nx - 1)
dy = (y_end - y_start) / (ny - 1)

# Time parameters
t_final = 0.40
dt = 0.005        # time step (chosen to satisfy CFL condition with max velocity ~2)
nt = int(t_final / dt)

# Create the grid
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)

# Initialize fields: u and v are 2D arrays
u = np.ones((nx, ny))
v = np.ones((nx, ny))

# Set initial conditions: u = v = 2 for 0.5 <= x <= 1 and 0.5 <= y <= 1; else 1
for i in range(nx):
    for j in range(ny):
        if 0.5 <= x[i] <= 1.0 and 0.5 <= y[j] <= 1.0:
            u[i, j] = 2.0
            v[i, j] = 2.0

# Enforce Dirichlet boundary conditions at t=0
u[0, :] = 1.0
u[-1, :] = 1.0
u[:, 0] = 1.0
u[:, -1] = 1.0

v[0, :] = 1.0
v[-1, :] = 1.0
v[:, 0] = 1.0
v[:, -1] = 1.0

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update interior points using first order upwind scheme
    # For u-equation
    u[1:-1, 1:-1] = un[1:-1, 1:-1] - dt * (
        un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) / dx +
        vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) / dy )
    
    # For v-equation
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] - dt * (
        un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) / dx +
        vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) / dy )
    
    # Enforce Dirichlet boundary conditions at every time step
    u[0, :] = 1.0
    u[-1, :] = 1.0
    u[:, 0] = 1.0
    u[:, -1] = 1.0
    
    v[0, :] = 1.0
    v[-1, :] = 1.0
    v[:, 0] = 1.0
    v[:, -1] = 1.0

# Save final time step solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Inviscid_Burgers.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Inviscid_Burgers.npy', v)