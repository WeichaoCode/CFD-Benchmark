import numpy as np

# Parameters
nx, ny = 41, 41
nt = 500
dx = 2.0 / (nx - 1)
dy = 2.0 / (ny - 1)
rho = 1.0
nu = 0.1
dt = 0.001
tol = 1e-4

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Boundary Conditions
def set_boundary(u, v, p):
    # u boundary conditions
    u[0, :] = 0
    u[-1, :] = 1
    u[:, 0] = 0
    u[:, -1] = 0

    # v boundary conditions
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0

    # Pressure boundary conditions
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]
    p[0, :] = p[1, :]
    p[-1, :] = 0

# Pressure Poisson equation solver
def pressure_poisson(p, b, dx, dy, tol=1e-4):
    pn = np.empty_like(p)
    iteration = 0
    while True:
        pn[:] = p
        p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
                         (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2) /
                        (2 * (dx**2 + dy**2)) -
                        dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1,1:-1])

        # Boundary conditions
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = 0

        # Check for convergence
        if np.max(np.abs(p - pn)) < tol:
            break
        iteration += 1
    return p

# Main time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Compute the source term
    b[1:-1,1:-1] = (rho * (1/dt *
                    ((un[1:-1,2:] - un[1:-1,0:-2]) / (2*dx) +
                     (vn[2:,1:-1] - vn[0:-2,1:-1]) / (2*dy)) -
                    ((un[1:-1,2:] - un[1:-1,0:-2]) / (2*dx))**2 -
                      2 * ((un[2:,1:-1] - un[0:-2,1:-1]) / (2*dy) *
                           (vn[1:-1,2:] - vn[1:-1,0:-2]) / (2*dx)) -
                    ((vn[2:,1:-1] - vn[0:-2,1:-1]) / (2*dy))**2))

    # Solve pressure Poisson equation
    p = pressure_poisson(p, b, dx, dy, tol)

    # Update velocity fields
    u[1:-1,1:-1] = (un[1:-1,1:-1] -
                    un[1:-1,1:-1] * dt / dx * (un[1:-1,1:-1] - un[1:-1,0:-2]) -
                    vn[1:-1,1:-1] * dt / dy * (un[1:-1,1:-1] - un[0:-2,1:-1]) -
                    dt / (2 * rho * dx) * (p[1:-1,2:] - p[1:-1,0:-2]) +
                    nu * (dt / dx**2 * (un[1:-1,2:] - 2 * un[1:-1,1:-1] + un[1:-1,0:-2]) +
                          dt / dy**2 * (un[2:,1:-1] - 2 * un[1:-1,1:-1] + un[0:-2,1:-1])))

    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                    un[1:-1,1:-1] * dt / dx * (vn[1:-1,1:-1] - vn[1:-1,0:-2]) -
                    vn[1:-1,1:-1] * dt / dy * (vn[1:-1,1:-1] - vn[0:-2,1:-1]) -
                    dt / (2 * rho * dy) * (p[2:,1:-1] - p[0:-2,1:-1]) +
                    nu * (dt / dx**2 * (vn[1:-1,2:] - 2 * vn[1:-1,1:-1] + vn[1:-1,0:-2]) +
                          dt / dy**2 * (vn[2:,1:-1] - 2 * vn[1:-1,1:-1] + vn[0:-2,1:-1])))

    # Apply boundary conditions
    set_boundary(u, v, p)

# Save final fields
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/u_2D_Navier_Stokes_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/v_2D_Navier_Stokes_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/p_2D_Navier_Stokes_Cavity.npy', p)