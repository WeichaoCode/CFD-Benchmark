import numpy as np

# Parameters
nx = 41  # number of points in x
ny = 41  # number of points in y
dx = 1.0/(nx-1)
dy = 1.0/(ny-1)
rho = 1.0
nu = 0.1
dt = 0.001
nsteps = 1000

# Initialize variables
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Set initial conditions
u[:, :] = 0
v[:, :] = 0
p[:, :] = 0

# Top lid velocity
u[-1, :] = 1.0

def pressure_poisson(p, b, dx, dy):
    pn = np.empty_like(p)
    for q in range(50):
        pn = p.copy()
        p[1:-1, 1:-1] = 0.25*(pn[1:-1, 2:] + pn[1:-1, :-2] + 
                              pn[2:, 1:-1] + pn[:-2, 1:-1] - 
                              dx*dy*b[1:-1, 1:-1])
        # Neumann boundary conditions
        p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 1
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 1
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        
    return p

# Main loop
for n in range(nsteps):
    un = u.copy()
    vn = v.copy()
    
    # Compute tentative velocity field
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                     dt/dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     dt/dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1]) +
                     nu*dt/dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) +
                     nu*dt/dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     dt/dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     dt/dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) +
                     nu*dt/dx**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                     nu*dt/dy**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1]))
    
    # Compute RHS of pressure Poisson equation
    b[1:-1, 1:-1] = rho*(1/dt * ((u[1:-1, 2:] - u[1:-1, :-2])/(2*dx) + 
                                 (v[2:, 1:-1] - v[:-2, 1:-1])/(2*dy)))
    
    # Solve pressure Poisson equation
    p = pressure_poisson(p, b, dx, dy)
    
    # Correct velocities
    u[1:-1, 1:-1] -= dt/(rho*dx) * (p[1:-1, 2:] - p[1:-1, :-2])
    v[1:-1, 1:-1] -= dt/(rho*dy) * (p[2:, 1:-1] - p[:-2, 1:-1])
    
    # Enforce boundary conditions
    u[0, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    u[-1, :] = 1    # Top lid
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0

# Save final solutions
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/u_Lid_Driven_Cavity.npy', u)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/v_Lid_Driven_Cavity.npy', v)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/p_Lid_Driven_Cavity.npy', p)