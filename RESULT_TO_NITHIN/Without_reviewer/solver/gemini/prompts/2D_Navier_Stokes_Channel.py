import numpy as np

def solve_cfd():
    # Parameters
    rho = 1.0
    nu = 0.1
    F = 1.0
    Lx = 2.0
    Ly = 2.0
    T = 5.0
    nx = 21
    ny = 21
    nt = 100
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = T / (nt - 1)

    # Initialize variables
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.zeros((nx, ny))

    # Functions for derivatives
    def laplacian(phi):
        return (np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) - 2 * phi) / dx**2 + \
               (np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - 2 * phi) / dy**2

    def dudx(phi):
        return (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)

    def dudy(phi):
        return (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dy)

    # Time loop
    for n in range(nt):
        # Solve momentum equations
        u_new = u + dt * (-u * dudx(u) - v * dudy(u) - (1 / rho) * dudx(p) + nu * laplacian(u) + F)
        v_new = v + dt * (-u * dudx(v) - v * dudy(v) - (1 / rho) * dudy(p) + nu * laplacian(v))

        # Solve pressure Poisson equation
        rhs = -rho * (dudx(u)**2 + 2 * dudy(u) * dudx(v) + dudy(v)**2)
        p_new = np.zeros((nx, ny))

        # Iterative solver for pressure Poisson equation
        for _ in range(50):
            p_new = 0.25 * (np.roll(p_new, 1, axis=0) + np.roll(p_new, -1, axis=0) +
                             np.roll(p_new, 1, axis=1) + np.roll(p_new, -1, axis=1) -
                             dx**2 * rhs)

            # Boundary conditions for pressure
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]

        # Update variables
        u = u_new
        v = v_new
        p = p_new

        # Periodic boundary conditions in x-direction
        u[0, :] = u[-1, :]
        u[-1, :] = u[0, :]
        v[0, :] = v[-1, :]
        v[-1, :] = v[0, :]
        p[0, :] = p[-1, :]
        p[-1, :] = p[0, :]

        # No-slip boundary conditions in y-direction
        u[:, 0] = 0
        u[:, -1] = 0
        v[:, 0] = 0
        v[:, -1] = 0

    # Save the final solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Navier_Stokes_Channel.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/v_2D_Navier_Stokes_Channel.npy', v)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Navier_Stokes_Channel.npy', p)

solve_cfd()