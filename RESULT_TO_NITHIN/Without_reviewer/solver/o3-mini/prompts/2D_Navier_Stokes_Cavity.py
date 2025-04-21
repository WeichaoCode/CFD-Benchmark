#!/usr/bin/env python3
import numpy as np

# Domain parameters
nx = 41
ny = 41
lx = 2.0
ly = 2.0
dx = lx / (nx - 1)
dy = ly / (ny - 1)

# Physical parameters
rho = 1.0
nu = 0.1

# Time stepping parameters
dt = 0.001
t_final = 10.0
nt = int(t_final/dt)

# Pressure Poisson solver iterations per time step
nit = 50

# Create grid
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)

# Initialize variables: u, v, p
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
p = np.zeros((nx, ny))

for n in range(nt):
    un = u.copy()
    vn = v.copy()
    pn = p.copy()
    
    # Compute source term for pressure Poisson equation
    # b = -( (du/dx)^2 + 2*(du/dy)*(dv/dx) + (dv/dy)^2 )
    b = np.zeros((nx, ny))
    # Use central differences for interior points
    b[1:-1,1:-1] = - ( ((un[2:,1:-1] - un[:-2,1:-1])/(2*dx))**2 
                      + 2 * ((un[1:-1,2:] - un[1:-1,:-2])/(2*dy))*((vn[2:,1:-1] - vn[:-2,1:-1])/(2*dx))
                      + ((vn[1:-1,2:] - vn[1:-1,:-2])/(2*dy))**2 )
    
    # Pressure Poisson equation solve
    for _ in range(nit):
        p_old = p.copy()
        p[1:-1,1:-1] = (((p_old[2:,1:-1] + p_old[:-2,1:-1]) * dy**2 +
                         (p_old[1:-1,2:] + p_old[1:-1,:-2]) * dx**2 -
                         b[1:-1,1:-1] * dx**2 * dy**2)
                        / (2*(dx**2+dy**2)))
        
        # Apply pressure boundary conditions
        # dp/dx = 0 at x = 0 and x = lx: p[0,:] = p[1,:], p[-1,:] = p[-2,:]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        # dp/dy = 0 at y = 0: p[:,0] = p[:,1]
        p[:, 0] = p[:, 1]
        # p = 0 at y = ly
        p[:, -1] = 0.0

    # Update velocity fields using the momentum equations
    # Interior points update for u and v.
    u[1:-1,1:-1] = (un[1:-1,1:-1] -
                     dt * (un[1:-1,1:-1]*((un[2:,1:-1]-un[:-2,1:-1])/(2*dx)) +
                           vn[1:-1,1:-1]*((un[1:-1,2:]-un[1:-1,:-2])/(2*dy))) -
                     dt * ( (p[2:,1:-1]-p[:-2,1:-1])/(2*rho*dx) ) +
                     dt * nu * ((un[2:,1:-1]-2*un[1:-1,1:-1]+un[:-2,1:-1])/(dx**2) +
                                (un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,:-2])/(dy**2)) )
    
    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                     dt * (un[1:-1,1:-1]*((vn[2:,1:-1]-vn[:-2,1:-1])/(2*dx)) +
                           vn[1:-1,1:-1]*((vn[1:-1,2:]-vn[1:-1,:-2])/(2*dy))) -
                     dt * ( (p[1:-1,2:]-p[1:-1,:-2])/(2*rho*dy) ) +
                     dt * nu * ((vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[:-2,1:-1])/(dx**2) +
                                (vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,:-2])/(dy**2)) )
    
    # Apply boundary conditions for velocity
    # No-slip conditions on all walls except the top (lid-driven)
    # Left and right boundaries (x=0 and x=lx)
    u[0, :] = 0.0
    u[-1, :] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0
    
    # Bottom wall (y=0)
    u[:, 0] = 0.0
    v[:, 0] = 0.0
    
    # Top wall (y=ly): lid-driven: u = 1, v = 0
    u[:, -1] = 1.0
    v[:, -1] = 0.0

# Save final solutions as .npy files (2D arrays)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Navier_Stokes_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Navier_Stokes_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Navier_Stokes_Cavity.npy', p)