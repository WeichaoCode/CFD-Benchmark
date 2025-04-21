#!/usr/bin/env python3
import numpy as np

# Parameters
nx = 41
ny = 41
lx = 1.0
ly = 1.0
dx = lx / (nx - 1)
dy = ly / (ny - 1)
rho = 1.0
nu = 0.1
dt = 0.001
nt = 5000            # number of time steps
nit = 50             # number of iterations for pressure Poisson

# Create grid
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)

# Initialize fields: u, v and p (2D arrays)
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Main time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Compute intermediate velocities u_star and v_star (without pressure gradient)
    u_star = un.copy()
    v_star = vn.copy()
    
    # Update interior points for u_star
    u_star[1:-1,1:-1] = (un[1:-1,1:-1] -
                          un[1:-1,1:-1] * dt/dx * (un[1:-1,1:-1] - un[1:-1,0:-2]) -
                          vn[1:-1,1:-1] * dt/dy * (un[1:-1,1:-1] - un[0:-2,1:-1]) +
                          nu * dt * ((un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]) / dx**2 +
                                     (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1]) / dy**2))
    
    # Update interior points for v_star
    v_star[1:-1,1:-1] = (vn[1:-1,1:-1] -
                          un[1:-1,1:-1] * dt/dx * (vn[1:-1,1:-1] - vn[1:-1,0:-2]) -
                          vn[1:-1,1:-1] * dt/dy * (vn[1:-1,1:-1] - vn[0:-2,1:-1]) +
                          nu * dt * ((vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,0:-2]) / dx**2 +
                                     (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[0:-2,1:-1]) / dy**2))
    
    # Enforce velocity boundary conditions on the intermediate velocity fields
    # Left, right and bottom walls: no-slip (u = 0, v = 0)
    u_star[:, 0] = 0
    u_star[:, -1] = 0
    u_star[0, :] = 0
    v_star[:, 0] = 0
    v_star[:, -1] = 0
    v_star[0, :] = 0
    # Top wall (driven lid): u = 1, v = 0
    u_star[-1, :] = 1.0
    v_star[-1, :] = 0

    # Solve pressure Poisson equation for p
    b = np.zeros((ny, nx))
    # Compute divergence of u_star and v_star
    b[1:-1,1:-1] = (rho/dt)*(((u_star[1:-1,2:] - u_star[1:-1,0:-2])/(2*dx)) +
                              ((v_star[2:,1:-1] - v_star[0:-2,1:-1])/(2*dy)))
    
    for it in range(nit):
        pn = p.copy()
        p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
                         (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2 -
                         b[1:-1,1:-1]*dx**2*dy**2) /
                        (2*(dx**2 + dy**2)))
        # Homogeneous Neumann BC for pressure (dp/dn = 0)
        p[:, -1] = p[:, -2]   # right wall
        p[:, 0] = p[:, 1]     # left wall
        p[-1, :] = p[-2, :]   # top wall
        p[0, :] = p[1, :]     # bottom wall
        # Set a reference pressure point for uniqueness
        p[0,0] = 0.0
    
    # Correct velocities with pressure gradient
    u[1:-1,1:-1] = (u_star[1:-1,1:-1] - dt/rho * ((p[1:-1,2:] - p[1:-1,0:-2]) / (2*dx)))
    v[1:-1,1:-1] = (v_star[1:-1,1:-1] - dt/rho * ((p[2:,1:-1] - p[0:-2,1:-1]) / (2*dy)))
    
    # Reapply velocity boundary conditions
    # No-slip on left, right, bottom walls:
    u[:,0] = 0
    u[:, -1] = 0
    u[0, :] = 0
    v[:,0] = 0
    v[:, -1] = 0
    v[0, :] = 0
    # Top wall (driven lid):
    u[-1, :] = 1.0
    v[-1, :] = 0

# Save final solutions as .npy files (each as a 2D array)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_Lid_Driven_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_Lid_Driven_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_Lid_Driven_Cavity.npy', p)