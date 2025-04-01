import numpy as np

# Parameters
nx = 51
ny = 51
nt = 500
c = 1
rho = 1
nu = 0.1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = 0.001
nit = 50

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Boundary conditions
u[-1, :] = 1  # Lid-driven top boundary

for _ in range(nt):
    un = u.copy()
    vn = v.copy()
    pn = p.copy()

    # Build up the RHS of the pressure Poisson equation
    b[1:-1,1:-1] = (rho * (1 / dt * 
                    ((un[1:-1,2:] - un[1:-1,0:-2]) / (2 * dx) + 
                     (vn[2:,1:-1] - vn[0:-2,1:-1]) / (2 * dy)) -
                    ((un[1:-1,2:] - un[1:-1,0:-2]) / (2 * dx))**2 -
                      2 * ((un[2:,1:-1] - un[0:-2,1:-1]) / (2 * dy) *
                           (vn[1:-1,2:] - vn[1:-1,0:-2]) / (2 * dx)) -
                    ((vn[2:,1:-1] - vn[0:-2,1:-1]) / (2 * dy))**2))

    # Pressure Poisson equation
    for _ in range(nit):
        pn = p.copy()
        p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
                          (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1,1:-1])

        # Pressure boundary conditions
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[-1, :] = 0         # p = 0 at y = 2

    # Update velocity fields
    u[1:-1,1:-1] = (un[1:-1,1:-1] -
                    un[1:-1,1:-1] * dt / dx * 
                   (un[1:-1,1:-1] - un[1:-1,0:-2]) -
                    vn[1:-1,1:-1] * dt / dy * 
                   (un[1:-1,1:-1] - un[0:-2,1:-1]) -
                    dt / (2 * rho * dx) * (p[1:-1,2:] - p[1:-1,0:-2]) +
                    nu * (dt / dx**2 * 
                   (un[1:-1,2:] - 2 * un[1:-1,1:-1] + un[1:-1,0:-2]) +
                   dt / dy**2 * 
                   (un[2:,1:-1] - 2 * un[1:-1,1:-1] + un[0:-2,1:-1])))

    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                    un[1:-1,1:-1] * dt / dx * 
                   (vn[1:-1,1:-1] - vn[1:-1,0:-2]) -
                    vn[1:-1,1:-1] * dt / dy * 
                   (vn[1:-1,1:-1] - vn[0:-2,1:-1]) -
                    dt / (2 * rho * dy) * (p[2:,1:-1] - p[0:-2,1:-1]) +
                    nu * (dt / dx**2 * 
                   (vn[1:-1,2:] - 2 * vn[1:-1,1:-1] + vn[1:-1,0:-2]) +
                   dt / dy**2 * 
                   (vn[2:,1:-1] - 2 * vn[1:-1,1:-1] + vn[0:-2,1:-1])))

    # Velocity boundary conditions
    u[0, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    u[-1, :] = 1

    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0

# Save the final fields
np.save('u.npy', u)
np.save('v.npy', v)
np.save('p.npy', p)