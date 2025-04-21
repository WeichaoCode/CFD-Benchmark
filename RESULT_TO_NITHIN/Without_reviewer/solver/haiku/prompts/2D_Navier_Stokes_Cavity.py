import numpy as np

# Grid parameters
nx = 41
ny = 41
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
dt = 0.001
nt = int(10.0/dt)

# Physical parameters
rho = 1.0
nu = 0.1

# Initialize fields
u = np.zeros((ny,nx))
v = np.zeros((ny,nx))
p = np.zeros((ny,nx))
b = np.zeros((ny,nx))

# Lid-driven cavity boundary condition
u[-1,:] = 1.0  # Top wall moving

def pressure_poisson(p, u, v, dx, dy, rho):
    pn = np.empty_like(p)
    
    for q in range(50):
        pn = p.copy()
        p[1:-1,1:-1] = ((pn[1:-1,2:] + pn[1:-1,:-2])*dy**2 + 
                        (pn[2:,1:-1] + pn[:-2,1:-1])*dx**2)/(2*(dx**2 + dy**2)) - \
                        rho*dx**2*dy**2/(2*(dx**2 + dy**2))*\
                        ((u[1:-1,2:] - u[1:-1,:-2])/(2*dx))**2 + \
                        2*((u[2:,1:-1] - u[:-2,1:-1])/(2*dy))*\
                        ((v[1:-1,2:] - v[1:-1,:-2])/(2*dx)) + \
                        ((v[2:,1:-1] - v[:-2,1:-1])/(2*dy))**2
                        
        # Boundary conditions
        p[-1,:] = 0  # Top boundary p = 0
        p[0,:] = p[1,:]  # Bottom boundary dp/dy = 0
        p[:,0] = p[:,1]  # Left boundary dp/dx = 0
        p[:,-1] = p[:,-2]  # Right boundary dp/dx = 0
        
    return p

# Time stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Solve pressure Poisson equation
    p = pressure_poisson(p, u, v, dx, dy, rho)
    
    # X-momentum equation
    u[1:-1,1:-1] = un[1:-1,1:-1] - \
                   dt*(un[1:-1,1:-1]*(un[1:-1,2:] - un[1:-1,:-2])/(2*dx) + \
                       vn[1:-1,1:-1]*(un[2:,1:-1] - un[:-2,1:-1])/(2*dy)) - \
                   dt/(rho)*(p[1:-1,2:] - p[1:-1,:-2])/(2*dx) + \
                   nu*dt*(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])/dx**2 + \
                   nu*dt*(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1])/dy**2
                   
    # Y-momentum equation
    v[1:-1,1:-1] = vn[1:-1,1:-1] - \
                   dt*(un[1:-1,1:-1]*(vn[1:-1,2:] - vn[1:-1,:-2])/(2*dx) + \
                       vn[1:-1,1:-1]*(vn[2:,1:-1] - vn[:-2,1:-1])/(2*dy)) - \
                   dt/(rho)*(p[2:,1:-1] - p[:-2,1:-1])/(2*dy) + \
                   nu*dt*(vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])/dx**2 + \
                   nu*dt*(vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1])/dy**2
    
    # Boundary conditions
    u[0,:] = 0
    u[:,-1] = 0
    u[:,0] = 0
    u[-1,:] = 1    # Moving lid
    v[0,:] = 0
    v[-1,:] = 0
    v[:,0] = 0
    v[:,-1] = 0

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Navier_Stokes_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/v_2D_Navier_Stokes_Cavity.npy', v) 
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/p_2D_Navier_Stokes_Cavity.npy', p)