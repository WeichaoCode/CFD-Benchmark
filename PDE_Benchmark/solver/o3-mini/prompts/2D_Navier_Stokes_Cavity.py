#!/usr/bin/env python3
import numpy as np

# Physical parameters
rho = 1.0
nu = 0.1
Lx = 2.0
Ly = 2.0
T = 10.0

# Numerical parameters
nx = 41
ny = 41
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = 0.001
nt = int(T / dt)
nit = 50  # number of iterations for pressure Poisson equation

# Create spatial grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize variables: arrays are 2D (ny x nx) where index 0 corresponds to y=0 and index -1 to y=Ly.
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

for n in range(nt):
    un = u.copy()
    vn = v.copy()
    pn = p.copy()
    
    # Build the RHS of the Pressure Poisson equation based on velocity derivatives.
    b[1:-1, 1:-1] = (rho * (1/dt * 
                   ((un[1:-1, 2:] - un[1:-1, 0:-2]) / (2*dx) + 
                    (vn[2:, 1:-1] - vn[0:-2, 1:-1]) / (2*dy)) -
                   ((un[1:-1, 2:] - un[1:-1, 0:-2]) / (2*dx))**2 -
                   2 * ((un[2:, 1:-1] - un[0:-2, 1:-1]) / (2*dy) *
                        (vn[1:-1, 2:] - vn[1:-1, 0:-2]) / (2*dx)) -
                        ((vn[2:, 1:-1] - vn[0:-2, 1:-1]) / (2*dy))**2))
    
    # Pressure Poisson equation
    for it in range(nit):
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                           (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2 -
                           b[1:-1, 1:-1] * dx**2 * dy**2) /
                          (2 * (dx**2 + dy**2)))
        # Apply pressure boundary conditions:
        # Left and right: dp/dx = 0
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        # Bottom: dp/dy = 0
        p[0, :] = p[1, :]
        # Top: p = 0
        p[-1, :] = 0
        pn = p.copy()
    
    # Update velocity field u
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt/(2*rho*dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * dt/(dx**2) * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     nu * dt/(dy**2) * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[0:-2, 1:-1]))
    
    # Update velocity field v
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt/dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt/(2*rho*dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * dt/(dx**2) * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     nu * dt/(dy**2) * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
    
    # Apply velocity boundary conditions
    # Left and right walls: u = 0, v = 0
    u[:, 0] = 0
    u[:, -1] = 0
    v[:, 0] = 0
    v[:, -1] = 0
    # Bottom wall (y=0): no-slip
    u[0, :] = 0
    v[0, :] = 0
    # Top wall (y=Ly): lid-driven condition: u = 1, v = 0
    u[-1, :] = 1
    v[-1, :] = 0

# Save the final solution variables as .npy files
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Navier_Stokes_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Navier_Stokes_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Navier_Stokes_Cavity.npy', p)