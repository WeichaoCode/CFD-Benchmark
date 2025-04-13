import numpy as np

# Parameters
nx, ny = 41, 41  # Grid points
lx, ly = 1.0, 1.0  # Domain size
dx, dy = lx / (nx - 1), ly / (ny - 1)
rho = 1.0
nu = 0.1
dt = 0.001
tolerance = 1e-5

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Boundary conditions
def apply_boundary_conditions(u, v, p):
    u[-1, :] = 1  # Top lid
    u[0, :] = 0  # Bottom wall
    u[:, 0] = 0  # Left wall
    u[:, -1] = 0  # Right wall

    v[-1, :] = 0  # Top lid
    v[0, :] = 0  # Bottom wall
    v[:, 0] = 0  # Left wall
    v[:, -1] = 0  # Right wall

    p[:, -1] = p[:, -2]  # dp/dx = 0 at right wall
    p[:, 0] = p[:, 1]  # dp/dx = 0 at left wall
    p[0, :] = p[1, :]  # dp/dy = 0 at bottom wall
    p[-1, :] = p[-2, :]  # dp/dy = 0 at top wall

# Pressure Poisson equation
def pressure_poisson(p, b):
    pn = np.empty_like(p)
    for _ in range(50):  # Iterations for Poisson equation
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        apply_boundary_conditions(u, v, p)

# Main loop
def cavity_flow():
    udiff = 1
    stepcount = 0
    while udiff > tolerance:
        un = u.copy()
        vn = v.copy()

        # Build up the RHS of the Poisson equation
        b[1:-1, 1:-1] = (rho * (1 / dt *
                                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                                 (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
                                2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                                ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2))

        # Pressure Poisson equation
        pressure_poisson(p, b)

        # Velocity field update
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx**2 *
                               (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                               dt / dy**2 *
                               (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx**2 *
                               (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                               dt / dy**2 *
                               (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

        apply_boundary_conditions(u, v, p)

        udiff = (np.sum((u - un)**2) + np.sum((v - vn)**2))**0.5
        stepcount += 1

    return u, v, p

# Run the simulation
u, v, p = cavity_flow()

# Save the results
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_Lid_Driven_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/v_Lid_Driven_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/p_Lid_Driven_Cavity.npy', p)