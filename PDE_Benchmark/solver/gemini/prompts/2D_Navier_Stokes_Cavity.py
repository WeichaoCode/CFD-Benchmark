import numpy as np

def solve_lid_driven_cavity():
    # Parameters
    rho = 1.0
    nu = 0.1
    nx = 41
    ny = 41
    nt = 500
    dt = 0.01
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    # Initialize arrays
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    u_new = np.zeros((ny, nx))
    v_new = np.zeros((ny, nx))
    p_new = np.zeros((ny, nx))

    # Boundary conditions
    u[-1, :] = 1.0  # Top boundary (lid)

    # Iteration
    for n in range(nt):
        # Solve for u and v
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u_new[i, j] = u[i, j] + dt * (
                    -u[i, j] * (u[i, j+1] - u[i, j-1]) / (2 * dx)
                    -v[i, j] * (u[i+1, j] - u[i-1, j]) / (2 * dy)
                    -(1 / rho) * (p[i, j+1] - p[i, j-1]) / (2 * dx)
                    + nu * ((u[i, j+1] - 2 * u[i, j] + u[i, j-1]) / (dx**2) + (u[i+1, j] - 2 * u[i, j] + u[i-1, j]) / (dy**2))
                )
                v_new[i, j] = v[i, j] + dt * (
                    -u[i, j] * (v[i, j+1] - v[i, j-1]) / (2 * dx)
                    -v[i, j] * (v[i+1, j] - v[i-1, j]) / (2 * dy)
                    -(1 / rho) * (p[i+1, j] - p[i-1, j]) / (2 * dy)
                    + nu * ((v[i, j+1] - 2 * v[i, j] + v[i, j-1]) / (dx**2) + (v[i+1, j] - 2 * v[i, j] + v[i-1, j]) / (dy**2))
                )

        # Boundary conditions for u and v
        u_new[0, :] = 0.0
        u_new[:, 0] = 0.0
        u_new[:, -1] = 0.0
        u_new[-1, :] = 1.0

        v_new[0, :] = 0.0
        v_new[:, 0] = 0.0
        v_new[:, -1] = 0.0
        v_new[-1, :] = 0.0

        # Solve for pressure (Poisson equation)
        for _ in range(50):  # Iterate to convergence
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    p_new[i, j] = 0.25 * (
                        p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1]
                        - rho * (
                            ((u_new[i, j+1] - u_new[i, j-1]) / (2 * dx))**2
                            + 2 * ((u_new[i+1, j] - u_new[i-1, j]) / (2 * dy)) * ((v_new[i, j+1] - v_new[i, j-1]) / (2 * dx))
                            + ((v_new[i+1, j] - v_new[i-1, j]) / (2 * dy))**2
                        ) * (dx**2)
                    )

            # Boundary conditions for pressure
            p_new[0, :] = p_new[1, :]  # dp/dy = 0 at y = 0
            p_new[-1, :] = 0.0  # p = 0 at y = 2
            p_new[:, 0] = p_new[:, 1]  # dp/dx = 0 at x = 0
            p_new[:, -1] = p_new[:, -2]  # dp/dx = 0 at x = 2

            p = p_new.copy()

        u = u_new.copy()
        v = v_new.copy()

    return u, v, p

if __name__ == "__main__":
    u, v, p = solve_lid_driven_cavity()
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Navier_Stokes_Cavity.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/v_2D_Navier_Stokes_Cavity.npy', v)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Navier_Stokes_Cavity.npy', p)