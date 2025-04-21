#!/usr/bin/env python3
import numpy as np

# Parameters
nx = 41
ny = 41
nt = 500        # number of time steps
nit = 50        # number of pressure iterations per time step
dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)
rho = 1.0
nu = 0.1
dt = 0.001

# Create grid
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)

# Initialize variables: u, v (velocity components) and p (pressure)
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Build the right hand side of the pressure Poisson equation: b
    b[1:-1, 1:-1] = rho * (1/dt * ((un[1:-1, 2:] - un[1:-1, 0:-2])/(2*dx) + 
                                     (vn[2:, 1:-1] - vn[0:-2, 1:-1])/(2*dy)) -
                           ((un[1:-1, 2:] - un[1:-1, 0:-2])/(2*dx))**2 -
                           2 * ((un[2:, 1:-1] - un[0:-2, 1:-1])/(2*dy) * (vn[1:-1, 2:] - vn[1:-1, 0:-2])/(2*dx)) -
                           ((vn[2:, 1:-1] - vn[0:-2, 1:-1])/(2*dy))**2)

    # Pressure Poisson equation (iterative Gauss-Seidel)
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) -
                         b[1:-1, 1:-1]*dx**2*dy**2) / (2*(dx**2+dy**2))
        # Pressure boundary conditions: homogeneous Neumann
        p[:, -1] = p[:, -2]    # dp/dx = 0 at right boundary
        p[:, 0]  = p[:, 1]     # dp/dx = 0 at left boundary
        p[-1, :] = p[-2, :]    # dp/dy = 0 at top boundary
        p[0, :]  = p[1, :]     # dp/dy = 0 at bottom boundary
        # Optionally set a reference value (Dirichlet) for uniqueness:
        p[0, 0] = 0.0

    # Update velocity field using the momentum equations
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt/(2*rho*dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt/dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                           dt/dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[0:-2, 1:-1])))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt/dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt/(2*rho*dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt/dx**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                           dt/dy**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    # Boundary conditions for u and v
    # Left and right boundaries: u = 0, v = 0
    u[:, 0] = 0; u[:, -1] = 0
    v[:, 0] = 0; v[:, -1] = 0
    # Bottom wall: u = 0, v = 0
    u[0, :] = 0; v[0, :] = 0
    # Top lid: u = 1 (driven lid), v = 0
    u[-1, :] = 1
    v[-1, :] = 0

# Save final solutions as .npy files
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_Lid_Driven_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_Lid_Driven_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_Lid_Driven_Cavity.npy', p)