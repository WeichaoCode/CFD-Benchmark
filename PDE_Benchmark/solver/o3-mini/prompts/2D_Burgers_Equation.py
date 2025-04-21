#!/usr/bin/env python3
import numpy as np

# Domain parameters
nx = 81             # number of grid points in x
ny = 81             # number of grid points in y
x_start, x_end = 0, 2
y_start, y_end = 0, 2
dx = (x_end - x_start) / (nx - 1)
dy = (y_end - y_start) / (ny - 1)

# Time parameters
t_final = 0.027
dt = 0.0005         # time step chosen to satisfy stability (may need tuning)
nt = int(t_final / dt)

# Physical parameter
nu = 0.01

# Create the grid
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
X, Y = np.meshgrid(x, y)

# Initialize u and v
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set initial conditions: u = v = 2 for 0.5 <= x,y <= 1 inside the domain.
u[np.logical_and(X >= 0.5, X <= 1) & np.logical_and(Y >= 0.5, Y <= 1)] = 2
v[np.logical_and(X >= 0.5, X <= 1) & np.logical_and(Y >= 0.5, Y <= 1)] = 2

# Time-stepping loop (unsteady simulation)
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update interior points for u
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                      dt * (un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) / dx +
                            vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) / dy) +
                      nu * dt * ((un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) / dx**2 +
                                 (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) / dy**2))
    
    # Update interior points for v
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                      dt * (un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) / dx +
                            vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) / dy) +
                      nu * dt * ((vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) / dx**2 +
                                 (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) / dy**2))
    
    # Enforce Dirichlet boundary conditions: u = v = 1 on all boundaries
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Save the final solution fields as .npy files (2D arrays)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Burgers_Equation.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Burgers_Equation.npy', v)