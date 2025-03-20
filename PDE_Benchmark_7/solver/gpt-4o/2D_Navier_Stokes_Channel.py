import numpy as np
import matplotlib.pyplot as plt

# Domain and grid parameters
nx, ny = 41, 41
nt = 10
dx, dy = 2.0 / (nx - 1), 2.0 / (ny - 1)
rho = 1.0  # Density
nu = 0.1   # Kinematic viscosity
F = 1.0    # Source term

# Stability condition for time step
sigma = 0.2
dt = sigma * min(dx, dy)**2 / nu

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

def build_up_b(u, v, dx, dy, rho):
    b = np.zeros_like(p)
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                           ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                            (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                           ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                             2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * 
                                  (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy):
    pn = np.empty_like(p)
    for _ in range(50):  # Iteration for the Poisson equation
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                         b[1:-1, 1:-1])

        # Periodic BCs in x
        p[1:-1, 0] = ((pn[1:-1, 1] + pn[1:-1, -1]) * dy**2 +
                      (pn[2:, 0] + pn[0:-2, 0]) * dx**2) / (2 * (dx**2 + dy**2))
        p[1:-1, -1] = p[1:-1, 0]

        # Neumann BCs in y
        p[0, :] = p[1, :]  # dp/dy = 0 at y=0
        p[-1, :] = p[-2, :]  # dp/dy = 0 at y=2
    return p

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    b = build_up_b(un, vn, dx, dy, rho)
    p = pressure_poisson(p, dx, dy)

    # Update velocity fields
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) *
                    (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
                     F * dt)

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt / (2 * rho * dy) *
                    (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    # Periodic BCs in x
    u[1:-1, 0] = u[1:-1, -1]
    u[1:-1, -1] = u[1:-1, 0]
    v[1:-1, 0] = v[1:-1, -1]
    v[1:-1, -1] = v[1:-1, 0]

    # No-slip BCs in y
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :] = 0

# Save results to .npy files
np.save('u.npy', u)
np.save('v.npy', v)
np.save('p.npy', p)

# Visualization
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(11,7), dpi=100)
plt.quiver(X, Y, u, v)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Velocity field')
plt.show()