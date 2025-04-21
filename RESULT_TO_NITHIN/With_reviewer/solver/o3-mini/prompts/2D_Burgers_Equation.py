#!/usr/bin/env python3
import numpy as np

# Parameters
nu = 0.01            # kinematic viscosity
Lx = 2.0             # domain size in x
Ly = 2.0             # domain size in y
nx = 81              # number of grid points in x
ny = 81              # number of grid points in y
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
t_final = 0.027      # final time
dt = 0.001           # time step
nt = int(t_final/dt) # number of time steps

# Generate grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize u and v fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set initial condition: interior region where 0.5 <= x,y <= 1, u,v = 2
# Note: we use meshgrid with indexing='ij' so that X[j,i] corresponds to x and Y[j,i] to y.
X, Y = np.meshgrid(x, y)
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2.0
v[mask] = 2.0

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Compute finite differences for interior points
    # Central differences for convection and diffusion terms.
    # u_x and u_y
    u_x = (un[1:-1, 2:] - un[1:-1, 0:-2]) / (2*dx)
    u_y = (un[2:, 1:-1] - un[0:-2, 1:-1]) / (2*dy)
    # u_xx and u_yy
    u_xx = (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, 0:-2]) / (dx**2)
    u_yy = (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[0:-2, 1:-1]) / (dy**2)
    
    # v_x and v_y
    v_x = (vn[1:-1, 2:] - vn[1:-1, 0:-2]) / (2*dx)
    v_y = (vn[2:, 1:-1] - vn[0:-2, 1:-1]) / (2*dy)
    # v_xx and v_yy
    v_xx = (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) / (dx**2)
    v_yy = (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) / (dy**2)

    # Update interior points for u
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                      dt * (un[1:-1, 1:-1] * u_x + vn[1:-1, 1:-1] * u_y) +
                      dt * nu * (u_xx + u_yy))
    
    # Update interior points for v
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                      dt * (un[1:-1, 1:-1] * v_x + vn[1:-1, 1:-1] * v_y) +
                      dt * nu * (v_xx + v_yy))
    
    # Apply Dirichlet boundary conditions: u = 1, v = 1 on all boundaries.
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Save final fields as .npy files (2D arrays)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Burgers_Equation.npy', u)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Burgers_Equation.npy', v)