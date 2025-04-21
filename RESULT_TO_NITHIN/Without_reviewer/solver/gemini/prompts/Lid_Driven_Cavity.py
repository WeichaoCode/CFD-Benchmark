import numpy as np

def solve_cavity_flow(nx=41, ny=41, nt=500, nu=0.1, rho=1.0, dt=0.001):
    """
    Solves the 2D cavity flow problem using a finite difference method.

    Args:
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        nt (int): Number of time steps.
        nu (float): Kinematic viscosity.
        rho (float): Fluid density.
        dt (float): Time step size.

    Returns:
        tuple: u, v, p (velocity components and pressure) at the final time step.
    """

    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    # Initialize variables
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    # Boundary conditions
    u[ny - 1, :] = 1.0  # Top lid

    def build_up_b(b, rho, dt, u, v, dx, dy):
        b[1:-1, 1:-1] = rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                        (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                                ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                                2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                                ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2)
        return b

    def pressure_poisson(p, dx, dy, b):
        pn = np.empty_like(p)
        pn[:] = p[:]

        for q in range(50):  # Number of iterations for pressure Poisson equation
            pn[:] = p[:]
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                               (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                              (2 * (dx**2 + dy**2)) -
                              dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

            # Neumann boundary conditions
            p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 1
            p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
            p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
            p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 1

        return p

    # Time loop
    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        # Compute intermediate velocities
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (un[1:-1, 1:-1] - un[0:-2, 1:-1]) +
                         nu * dt / dx**2 *
                         (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         nu * dt / dy**2 *
                         (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) +
                         nu * dt / dx**2 *
                         (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                         nu * dt / dy**2 *
                         (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))

        # Boundary conditions for intermediate velocities
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        v[-1, :] = 0
        v[0, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

        # Build RHS of pressure Poisson equation
        b = build_up_b(b, rho, dt, u, v, dx, dy)

        # Solve pressure Poisson equation
        p = pressure_poisson(p, dx, dy, b)

        # Correct velocities
        u[1:-1, 1:-1] = u[1:-1, 1:-1] - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
        v[1:-1, 1:-1] = v[1:-1, 1:-1] - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])

        # Boundary conditions for corrected velocities
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        v[-1, :] = 0
        v[0, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
        u[ny - 1, :] = 1.0

    return u, v, p


if __name__ == '__main__':
    # Problem parameters
    nx = 41
    ny = 41
    nt = 500
    nu = 0.1
    rho = 1.0
    dt = 0.001

    # Solve the cavity flow problem
    u, v, p = solve_cavity_flow(nx, ny, nt, nu, rho, dt)

    # Save the results
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_Lid_Driven_Cavity.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/v_Lid_Driven_Cavity.npy', v)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_Lid_Driven_Cavity.npy', p)