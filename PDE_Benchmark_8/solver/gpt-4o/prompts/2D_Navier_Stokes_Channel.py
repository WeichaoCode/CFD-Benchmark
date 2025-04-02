import numpy as np

# Parameters
nx, ny = 41, 41  # number of grid points
nt = 100  # number of time steps
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = 0.001  # time step size
rho = 1
nu = 0.1
F = 1

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Helper functions
def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                   ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + 
                    (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                   ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                     2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                          (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                     ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    for q in range(nt):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                         b[1:-1, 1:-1])

        # Periodic BC Pressure in x-direction
        p[:, -1] = p[:, 0]
        p[:, 0] = p[:, -1]

        # Neumann BC Pressure in y-direction
        p[-1, :] = p[-2, :]
        p[0, :] = p[1, :]

    return p

# Time-stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    b = build_up_b(b, rho, dt, u, v, dx, dy)
    p = pressure_poisson(p, dx, dy, b)

    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + F * dt)

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

    # Periodic BC u,v in x-direction
    u[:, -1] = u[:, 0]
    u[:, 0] = u[:, -1]
    v[:, -1] = v[:, 0]
    v[:, 0] = v[:, -1]

    # No-slip BC u,v in y-direction
    u[-1, :] = 0
    u[0, :] = 0
    v[-1, :] = 0
    v[0, :] = 0

# Save the final results
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts/u_2D_Navier_Stokes_Channel.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts/v_2D_Navier_Stokes_Channel.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts/p_2D_Navier_Stokes_Channel.npy', p)