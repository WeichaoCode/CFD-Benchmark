import numpy as np

# Parameters
rho = 1.0
nu = 0.1
F = 1.0
Lx, Ly = 2.0, 2.0
Nx, Ny = 64, 64
dx = Lx / Nx
dy = Ly / Ny
dt = 0.001
nt = int(0.1 / dt)

# Initialize fields
u = np.zeros((Ny, Nx))
v = np.zeros((Ny, Nx))
p = np.zeros((Ny, Nx))

# Helper functions
def apply_boundary_conditions(u, v, p):
    # Periodic boundary conditions in x
    u[:,0] = u[:,-2]
    u[:,-1] = u[:,1]
    v[:,0] = v[:,-2]
    v[:,-1] = v[:,1]
    p[:,0] = p[:,-2]
    p[:,-1] = p[:,1]
    
    # No-slip boundary conditions in y
    u[0,:] = 0
    u[-1,:] = 0
    v[0,:] = 0
    v[-1,:] = 0
    
    # Neumann boundary conditions for pressure in y
    p[0,:] = p[1,:]
    p[-1,:] = p[-2,:]
    return u, v, p

def pressure_poisson(p, b, dx, dy, iterations=50):
    pn = np.empty_like(p)
    for _ in range(iterations):
        pn[:,:] = p[:,:]
        p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
                         (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2) -
                        b * dx**2 * dy**2) / (2 * (dx**2 + dy**2))
        # Periodic BCs
        p[:,0] = p[:,-2]
        p[:,-1] = p[:,1]
        # Neumann BCs for pressure in y
        p[0,:] = p[1,:]
        p[-1,:] = p[-2,:]
    return p

# Time-stepping loop
for _ in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Compute derivatives
    du_dx = (un[1:-1,2:] - un[1:-1,0:-2]) / (2*dx)
    du_dy = (un[2:,1:-1] - un[0:-2,1:-1]) / (2*dy)
    dv_dx = (vn[1:-1,2:] - vn[1:-1,0:-2]) / (2*dx)
    dv_dy = (vn[2:,1:-1] - vn[0:-2,1:-1]) / (2*dy)
    
    d2u_dx2 = (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]) / dx**2
    d2u_dy2 = (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1]) / dy**2
    d2v_dx2 = (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,0:-2]) / dx**2
    d2v_dy2 = (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[0:-2,1:-1]) / dy**2
    
    # External force
    F_ext = F
    
    # Compute RHS for pressure Poisson
    b = rho * ((du_dx)**2 + 2*(du_dy * dv_dx) + (dv_dy)**2)
    
    # Solve pressure Poisson
    p = pressure_poisson(p, b, dx, dy)
    
    # Compute pressure gradients
    dp_dx = (p[1:-1,2:] - p[1:-1,0:-2]) / (2*dx)
    dp_dy = (p[2:,1:-1] - p[0:-2,1:-1]) / (2*dy)
    
    # Update velocities
    u[1:-1,1:-1] = (un[1:-1,1:-1] -
                     dt * (un[1:-1,1:-1] * du_dx + vn[1:-1,1:-1] * du_dy) -
                     dt * (1/rho) * dp_dx +
                     dt * nu * (d2u_dx2 + d2u_dy2) + dt * F_ext)
    
    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                     dt * (un[1:-1,1:-1] * dv_dx + vn[1:-1,1:-1] * dv_dy) -
                     dt * (1/rho) * dp_dy +
                     dt * nu * (d2v_dx2 + d2v_dy2))
    
    # Apply boundary conditions
    u, v, p = apply_boundary_conditions(u, v, p)

# Save final results
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/u_2D_Navier_Stokes_Channel.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/v_2D_Navier_Stokes_Channel.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/p_2D_Navier_Stokes_Channel.npy', p)