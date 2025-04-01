import numpy as np

# Parameters
nx, ny = 41, 41
nt = 10
dx = 2 / (nx -1)
dy = 2 / (ny -1)
rho = 1
nu = 0.1
F = 1
dt = 0.001
nit = 50  # Number of iterations for pressure Poisson

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Define periodic boundary conditions in x-direction
def apply_periodic_bc(var):
    var[:,0] = var[:, -2]
    var[:,-1] = var[:,1]

# Define no-slip boundary conditions in y-direction
def apply_no_slip_bc(var):
    var[0,:] = 0
    var[-1,:] = 0

# Pressure Poisson equation
def pressure_poisson(p, b):
    pn = np.empty_like(p)
    for q in range(nit):
        pn = p.copy()
        p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
                          (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1,1:-1])

        # Pressure boundary conditions
        p[:, -1] = p[:,1]  # Periodic BC
        p[:,0] = p[:,-2]   # Periodic BC
        p[0,:] = p[1,:]    # dp/dy = 0 at y=0
        p[-1,:] = p[-2,:]  # dp/dy = 0 at y=2
    return p

# Time-stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Compute RHS of pressure Poisson equation
    b[1:-1,1:-1] = (rho * (1/dt *
                    ((un[1:-1,2:] - un[1:-1,0:-2]) / (2 * dx) +
                     (vn[2:,1:-1] - vn[0:-2,1:-1]) / (2 * dy)) -
                     ((un[1:-1,2:] - un[1:-1,0:-2]) / (2 * dx))**2 -
                       2 * ((un[2:,1:-1] - un[0:-2,1:-1]) / (2 * dy) *
                            (vn[1:-1,2:] - vn[1:-1,0:-2]) / (2 * dx)) -
                     ((vn[2:,1:-1] - vn[0:-2,1:-1]) / (2 * dy))**2))

    # Solve pressure Poisson
    p = pressure_poisson(p, b)

    # Update velocity field
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

    # Apply boundary conditions
    apply_periodic_bc(u)
    apply_periodic_bc(v)
    apply_periodic_bc(p)
    apply_no_slip_bc(u)
    apply_no_slip_bc(v)

# Save final results
np.save('u.npy', u)
np.save('v.npy', v)
np.save('p.npy', p)