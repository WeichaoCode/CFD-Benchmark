import numpy as np

# Parameters
nx, ny = 50, 50  # number of grid points
lx, ly = 1.0, 1.0  # domain size
dx, dy = lx / (nx - 1), ly / (ny - 1)  # grid spacing
rho = 1.0  # density
nu = 0.1  # kinematic viscosity
dt = 0.001  # time step
tolerance = 1e-5  # convergence tolerance
max_iter = 10000  # maximum number of iterations

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Boundary conditions
u[-1, :] = 1  # top lid driven wall

def build_up_b(b, u, v, dx, dy, dt):
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                   ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                    (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                   ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                     2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                          (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                     ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

def pressure_poisson(p, dx, dy):
    pn = np.empty_like(p)
    for q in range(max_iter):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Neumann boundary conditions for pressure
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 2

        # Check for convergence
        if np.sum(np.abs(p - pn)) < tolerance:
            break

def cavity_flow(u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    for n in range(max_iter):
        un = u.copy()
        vn = v.copy()

        build_up_b(b, u, v, dx, dy, dt)
        pressure_poisson(p, dx, dy)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Boundary conditions
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1  # lid velocity is 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

        # Check for convergence
        if np.sum(np.abs(u - un)) < tolerance and np.sum(np.abs(v - vn)) < tolerance:
            break

cavity_flow(u, v, dt, dx, dy, p, rho, nu)

# Save the final results
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_Lid_Driven_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/v_Lid_Driven_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/p_Lid_Driven_Cavity.npy', p)