import numpy as np

# Parameters
nx, ny = 41, 41
nt = 10
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
rho = 1
nu = 0.1
F = 1
dt = 0.001

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Pressure Poisson solver parameters
pn = np.empty_like(p)
nit = 50

for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Build up the source term for the pressure Poisson equation
    b[1:-1,1:-1] = (rho * (1/dt * 
                    ((un[1:-1,2:] - un[1:-1,0:-2]) / (2*dx) + 
                     (vn[2:,1:-1] - vn[0:-2,1:-1]) / (2*dy)) -
                    ((un[1:-1,2:] - un[1:-1,0:-2]) / (2*dx))**2 -
                      2 * ((un[2:,1:-1] - un[0:-2,1:-1]) / (2*dy) *
                           (vn[1:-1,2:] - vn[1:-1,0:-2]) / (2*dx)) -
                    ((vn[2:,1:-1] - vn[0:-2,1:-1]) / (2*dy))**2))
    
    # Pressure Poisson equation
    pn = p.copy()
    for q in range(nit):
        p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
                          (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1,1:-1])

        # Boundary conditions for pressure
        p[:, -1] = p[:,1]    # Periodic BC in x
        p[:, 0] = p[:, -2]   # Periodic BC in x
        p[0, :] = p[1, :]    # Neumann BC at y=0
        p[-1, :] = p[-2, :]  # Neumann BC at y=2
        pn = p.copy()
    
    # Update velocity fields
    u[1:-1,1:-1] = (un[1:-1,1:-1] -
                    un[1:-1,1:-1] * dt / dx * (un[1:-1,1:-1] - un[1:-1,0:-2]) -
                    vn[1:-1,1:-1] * dt / dy * (un[1:-1,1:-1] - un[0:-2,1:-1]) -
                    dt / (2 * rho * dx) * (p[1:-1,2:] - p[1:-1,0:-2]) +
                    nu * (dt / dx**2 * (un[1:-1,2:] - 2 * un[1:-1,1:-1] + un[1:-1,0:-2]) +
                           dt / dy**2 * (un[2:,1:-1] - 2 * un[1:-1,1:-1] + un[0:-2,1:-1])) +
                    F * dt)
    
    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                    un[1:-1,1:-1] * dt / dx * (vn[1:-1,1:-1] - vn[1:-1,0:-2]) -
                    vn[1:-1,1:-1] * dt / dy * (vn[1:-1,1:-1] - vn[0:-2,1:-1]) -
                    dt / (2 * rho * dy) * (p[2:,1:-1] - p[0:-2,1:-1]) +
                    nu * (dt / dx**2 * (vn[1:-1,2:] - 2 * vn[1:-1,1:-1] + vn[1:-1,0:-2]) +
                           dt / dy**2 * (vn[2:,1:-1] - 2 * vn[1:-1,1:-1] + vn[0:-2,1:-1])))

    # Apply periodic boundary conditions in x
    u[:,0] = u[:,-2]
    u[:,-1] = u[:,1]
    v[:,0] = v[:,-2]
    v[:,-1] = v[:,1]
    
    # Apply no-slip boundary conditions at y boundaries
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :] = 0

# Save the final fields
np.save('u.npy', u)
np.save('v.npy', v)
np.save('p.npy', p)