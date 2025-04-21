#!/usr/bin/env python3
import numpy as np

# Domain and simulation parameters
nx = 41
ny = 41
lx = 1.0
ly = 1.0
dx = lx / (nx - 1)
dy = ly / (ny - 1)
nt = 500          # number of time steps
nit = 50          # number of iterations for Pressure Poisson equation per time step
dt = 0.001        # time step size

# Physical parameters
rho = 1.0         # fluid density
nu = 0.1          # kinematic viscosity

# Initialize fields: u, v, and p
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Time stepping loop (unsteady; final solution at final time step)
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Compute intermediate velocity field u*, v* (without pressure correction)
    # Use central differences for spatial derivatives and explicit time stepping.
    u_inner = un[1:-1, 1:-1]
    v_inner = vn[1:-1, 1:-1]
    
    # Convection terms
    du_dx = (un[1:-1, 2:] - un[1:-1, 0:-2]) / (2*dx)
    du_dy = (un[2:, 1:-1] - un[0:-2, 1:-1]) / (2*dy)
    dv_dx = (vn[1:-1, 2:] - vn[1:-1, 0:-2]) / (2*dx)
    dv_dy = (vn[2:, 1:-1] - vn[0:-2, 1:-1]) / (2*dy)
    
    # Diffusion terms using second order central differences
    d2u_dx2 = (un[1:-1, 2:] - 2*u_inner + un[1:-1, 0:-2]) / dx**2
    d2u_dy2 = (un[2:, 1:-1] - 2*u_inner + un[0:-2, 1:-1]) / dy**2
    d2v_dx2 = (vn[1:-1, 2:] - 2*v_inner + vn[1:-1, 0:-2]) / dx**2
    d2v_dy2 = (vn[2:, 1:-1] - 2*v_inner + vn[0:-2, 1:-1]) / dy**2
    
    u_star = u_inner + dt * (
                - u_inner * du_dx - v_inner * du_dy +
                nu * (d2u_dx2 + d2u_dy2)
             )
    v_star = v_inner + dt * (
                - u_inner * dv_dx - v_inner * dv_dy +
                nu * (d2v_dx2 + d2v_dy2)
             )
    
    # Temporary arrays for u_star and v_star placed in the interior
    u_temp = un.copy()
    v_temp = vn.copy()
    u_temp[1:-1, 1:-1] = u_star
    v_temp[1:-1, 1:-1] = v_star
    
    # Enforce boundary conditions on u_star and v_star (velocity BCs before pressure correction)
    # No-slip on all walls except top lid:
    u_temp[0, :] = 0.0       # bottom wall
    u_temp[:, 0] = 0.0       # left wall
    u_temp[:, -1] = 0.0      # right wall
    u_temp[-1, :] = 1.0      # top wall (driven lid)
    
    v_temp[0, :] = 0.0
    v_temp[-1, :] = 0.0
    v_temp[:, 0] = 0.0
    v_temp[:, -1] = 0.0
    
    # Pressure Poisson equation: iterative solution for p (using finite difference)
    pn = p.copy()
    for it in range(nit):
        pn_old = p.copy()
        # Interior points update using 5-point Laplacian
        p[1:-1, 1:-1] = ((pn_old[1:-1, 2:] + pn_old[1:-1, 0:-2]) * dy**2 +
                         (pn_old[2:, 1:-1] + pn_old[0:-2, 1:-1]) * dx**2 -
                         rho * dx**2 * dy**2 / dt *
                         ((u_temp[1:-1, 2:] - u_temp[1:-1, 0:-2]) / (2*dx) +
                          (v_temp[2:, 1:-1] - v_temp[0:-2, 1:-1]) / (2*dy))
                        ) / (2*(dx**2 + dy**2))
        
        # Pressure boundary conditions
        # Homogeneous Neumann: dp/dn=0 implemented as p at boundary equals neighbor.
        p[:, -1] = p[:, -2]   # right wall
        p[:, 0] = p[:, 1]     # left wall
        p[-1, :] = p[-2, :]   # top wall
        p[0, :] = p[1, :]     # bottom wall
        # Optionally fix one pressure point to zero for uniqueness (reference point)
        p[0, 0] = 0.0
        
    # Update velocity with pressure gradient correction
    u[1:-1, 1:-1] = u_temp[1:-1, 1:-1] - dt / (rho * 2*dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
    v[1:-1, 1:-1] = v_temp[1:-1, 1:-1] - dt / (rho * 2*dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
    
    # Enforce velocity boundary conditions after pressure correction
    u[0, :] = 0.0          # bottom wall
    u[:, 0] = 0.0          # left wall
    u[:, -1] = 0.0         # right wall
    u[-1, :] = 1.0         # top wall (driven lid)
    
    v[0, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    v[-1, :] = 0.0

# Save final time step solution fields as .npy files.
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/u_Lid_Driven_Cavity.npy', u)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/v_Lid_Driven_Cavity.npy', v)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/p_Lid_Driven_Cavity.npy', p)