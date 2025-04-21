#!/usr/bin/env python3
import numpy as np

# Parameters
nx = 41
ny = 41
lx = 2.0
ly = 2.0
dx = lx / (nx - 1)
dy = ly / (ny - 1)
t_final = 10.0
dt = 0.001         # reduced time step for stability
nt = int(t_final / dt)  # number of time steps to reach final time
rho = 1.0          # fluid density
nu = 0.1           # kinematic viscosity
nit = 50           # iterations for pressure Poisson equation per time step

# Create grid
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)

# Initialize fields: arrays are of shape (ny, nx)
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Main time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Compute derivatives for interior points using central differences on interior (indices 1:-1)
    du_dx = (un[1:-1, 2:] - un[1:-1, 0:-2]) / (2 * dx)  # shape (ny-2, nx-2)
    du_dy = (un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy)    # shape (ny-2, nx-2)
    dv_dx = (vn[1:-1, 2:] - vn[1:-1, 0:-2]) / (2 * dx)    # shape (ny-2, nx-2)
    dv_dy = (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)      # shape (ny-2, nx-2)
    
    # Compute source term for the pressure Poisson equation on interior points
    b[1:-1, 1:-1] = - (du_dx**2 + 2 * du_dy * dv_dx + dv_dy**2)
    
    # Pressure Poisson equation iterative solve for interior pressure points
    for it in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2 -
                          b[1:-1, 1:-1] * dx**2 * dy**2) /
                         (2 * (dx**2 + dy**2)))
        # Apply pressure boundary conditions
        p[:, 0] = p[:, 1]      # dp/dx = 0 at x = 0
        p[:, -1] = p[:, -2]    # dp/dx = 0 at x = lx
        p[0, :] = p[1, :]      # dp/dy = 0 at y = 0
        p[-1, :] = 0.0         # p = 0 at y = ly

    # Update velocity field for interior points
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                     dt * (
                         - un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) / dx
                         - vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) / dy
                         - (1 / rho) * (p[1:-1, 2:] - p[1:-1, 0:-2]) / (2 * dx)
                         + nu * ((un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) / dx**2 +
                                 (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) / dy**2)
                     ))
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] +
                     dt * (
                         - un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) / dx
                         - vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) / dy
                         - (1 / rho) * (p[2:, 1:-1] - p[0:-2, 1:-1]) / (2 * dy)
                         + nu * ((vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) / dx**2 +
                                 (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) / dy**2)
                     ))
    
    # Enforce velocity boundary conditions
    # Top boundary (lid-driven): u = 1, v = 0 at y = ly
    u[-1, :] = 1.0
    v[-1, :] = 0.0
    # Bottom boundary (no-slip): u = 0, v = 0 at y = 0
    u[0, :] = 0.0
    v[0, :] = 0.0
    # Left and right walls (no-slip): u = 0, v = 0 at x = 0 and x = lx
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0

# Save the final solution fields as .npy files
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Navier_Stokes_Cavity.npy', u)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Navier_Stokes_Cavity.npy', v)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Navier_Stokes_Cavity.npy', p)