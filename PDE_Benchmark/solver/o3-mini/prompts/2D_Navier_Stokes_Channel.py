#!/usr/bin/env python3
import numpy as np

# Parameters
nx = 41
ny = 41
Lx = 2.0
Ly = 2.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
t_final = 5.0
dt = 0.01
nt = int(t_final/dt)
rho = 1.0
nu = 0.1
F = 1.0

# Pressure Poisson parameters
nit = 50  # number of iterations for pressure Poisson

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize fields: u, v, p (2D arrays shape: ny x nx)
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    pn = p.copy()
    
    # Compute derivatives needed for momentum equations using central differences
    # Use periodic BC in x-direction via np.roll. For y, update only interior points.
    # Compute x-derivatives for u and v (full arrays, periodic in x)
    u_x = (np.roll(un, -1, axis=1) - np.roll(un, 1, axis=1))/(2*dx)
    v_x = (np.roll(vn, -1, axis=1) - np.roll(vn, 1, axis=1))/(2*dx)
    
    # For y-derivatives, only update interior rows (1 to ny-2)
    u_y = np.zeros_like(u)
    v_y = np.zeros_like(v)
    u_y[1:-1, :] = (un[2:, :] - un[:-2, :])/(2*dy)
    v_y[1:-1, :] = (vn[2:, :] - vn[:-2, :])/(2*dy)
    
    # Second derivatives for diffusion (only interior rows for y)
    u_xx = (np.roll(un, -1, axis=1) - 2*un + np.roll(un, 1, axis=1))/(dx*dx)
    u_yy = np.zeros_like(u)
    u_yy[1:-1, :] = (un[2:, :] - 2*un[1:-1, :] + un[:-2, :])/(dy*dy)
    
    v_xx = (np.roll(vn, -1, axis=1) - 2*vn + np.roll(vn, 1, axis=1))/(dx*dx)
    v_yy = np.zeros_like(v)
    v_yy[1:-1, :] = (vn[2:, :] - 2*vn[1:-1, :] + vn[:-2, :])/(dy*dy)
    
    # Pressure gradients (using periodic in x, central in y for interior)
    p_x = (np.roll(pn, -1, axis=1) - np.roll(pn, 1, axis=1))/(2*dx)
    p_y = np.zeros_like(p)
    p_y[1:-1, :] = (pn[2:, :] - pn[:-2, :])/(2*dy)
    
    # Update momentum equations for interior points (y: 1 to ny-2)
    u[1:-1, :] = un[1:-1, :] + dt * (
         - un[1:-1, :]*u_x[1:-1, :] - vn[1:-1, :]*u_y[1:-1, :]
         - p_x[1:-1, :] / rho
         + nu*(u_xx[1:-1, :] + u_yy[1:-1, :])
         + F
         )
    
    v[1:-1, :] = vn[1:-1, :] + dt * (
         - un[1:-1, :]*v_x[1:-1, :] - vn[1:-1, :]*v_y[1:-1, :]
         - p_y[1:-1, :] / rho
         + nu*(v_xx[1:-1, :] + v_yy[1:-1, :])
         )
    
    # Enforce boundary conditions for u and v in y (no-slip: u=v=0)
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :] = 0
    # x boundaries are periodic; already handled via np.roll in derivative computations.
    
    # Pressure Poisson Equation: solve for p with Neumann BC in y and periodic in x
    # Compute right-hand side based on updated velocities
    rhs = np.zeros((ny-2, nx))
    u_x_in = (np.roll(u, -1, axis=1)[1:-1, :] - np.roll(u, 1, axis=1)[1:-1, :])/(2*dx)
    u_y_in = (u[2:, :] - u[:-2, :])/(2*dy)
    v_x_in = (np.roll(v, -1, axis=1)[1:-1, :] - np.roll(v, 1, axis=1)[1:-1, :])/(2*dx)
    v_y_in = (v[2:, :] - v[:-2, :])/(2*dy)
    rhs = - (u_x_in**2 + 2*u_y_in*v_x_in + v_y_in**2)
    
    for it in range(nit):
        p_old = p.copy()
        # Update pressure for interior points y=1:-1
        p[1:-1, :] = (((np.roll(p, -1, axis=1)[1:-1, :] + np.roll(p, 1, axis=1)[1:-1, :])*(dy**2) +
                       (p[2:, :] + p[:-2, :])*(dx**2) -
                       rhs*(dx**2)*(dy**2))
                      / (2*(dx**2+dy**2)))
        # Enforce periodic BC in x for pressure automatically via np.roll
        # Enforce Neumann BC in y: dp/dy = 0 at y=0 and y=Ly
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        
        # Optional: check residual if desired (omitted)
    
# Save the final solution fields as .npy files
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Navier_Stokes_Channel.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Navier_Stokes_Channel.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Navier_Stokes_Channel.npy', p)