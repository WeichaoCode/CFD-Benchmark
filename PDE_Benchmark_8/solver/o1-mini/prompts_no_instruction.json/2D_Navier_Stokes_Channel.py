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
un = np.zeros_like(u)
vn = np.zeros_like(v)
pn = np.zeros_like(p)

# Pressure Poisson solver
def build_up_b(rho, dt, u, v, dx, dy):
    b = np.zeros_like(p)
    b[1:-1,1:-1] = (rho * (1/dt * 
                  ((u[1:-1,2:] - u[1:-1,0:-2]) / (2*dx) + 
                   (v[2:,1:-1] - v[0:-2,1:-1]) / (2*dy)) ) -
                  ((u[1:-1,2:] - u[1:-1,0:-2]) / (2*dx))**2 -
                  2 * ((u[2:,1:-1] - u[0:-2,1:-1]) / (2*dy) *
                       (v[1:-1,2:] - v[1:-1,0:-2]) / (2*dx)) -
                  ((v[2:,1:-1] - v[0:-2,1:-1]) / (2*dy))**2)
    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    pn = p.copy()
    for q in range(50):
        pn = p.copy()
        p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
                         (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2) /
                        (2 * (dx**2 + dy**2)) -
                        dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1,1:-1])

        # Boundary conditions
        p[:, -1] = p[:,0]  # Periodic BC in x
        p[:,0] = p[:,-2]
        p[0,:] = p[1,:]
        p[-1,:] = p[-2,:]
    return p

# Time-stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    b = build_up_b(rho, dt, u, v, dx, dy)
    p = pressure_poisson(p, dx, dy, b)

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
                    (un[2:,1:-1] - 2 * un[1:-1,1:-1] + un[0:-2,1:-1])) + 
                     F * dt)

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

    # Boundary conditions
    # Periodic BC in x
    u[:,0] = u[:,-2]
    u[:,-1] = u[:,1]
    v[:,0] = v[:,-2]
    v[:,-1] = v[:,1]
    p[:,0] = p[:,-2]
    p[:,-1] = p[:,1]

    # No-slip BC in y
    u[0,:] = 0
    u[-1,:] = 0
    v[0,:] = 0
    v[-1,:] = 0

    # dp/dy = 0 at y=0 and y=2
    p[0,:] = p[1,:]
    p[-1,:] = p[-2,:]

# Save the final results
np.save('u.npy', u)
np.save('v.npy', v)
np.save('p.npy', p)