import numpy as np

def solve_cfd():
    # Parameters
    nu = 0.01
    Lx = 2.0
    Ly = 2.0
    T = 0.027
    nx = 50
    ny = 50
    nt = 50

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = T / (nt - 1)

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Initialize u and v
    u = np.ones((nx, ny))
    v = np.ones((nx, ny))

    # Initial conditions
    for i in range(nx):
        for j in range(ny):
            if 0.5 <= x[i] <= 1.0 and 0.5 <= y[j] <= 1.0:
                u[i, j] = 2.0
                v[i, j] = 2.0

    # Boundary conditions
    u[:, 0] = 1.0
    u[:, -1] = 1.0
    u[0, :] = 1.0
    u[-1, :] = 1.0

    v[:, 0] = 1.0
    v[:, -1] = 1.0
    v[0, :] = 1.0
    v[-1, :] = 1.0

    # Time loop
    for n in range(nt):
        u_old = u.copy()
        v_old = v.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i, j] = u_old[i, j] + dt * (
                    nu * ((u_old[i+1, j] - 2*u_old[i, j] + u_old[i-1, j]) / dx**2 + (u_old[i, j+1] - 2*u_old[i, j] + u_old[i, j-1]) / dy**2)
                    - u_old[i, j] * (u_old[i+1, j] - u_old[i-1, j]) / (2*dx)
                    - v_old[i, j] * (u_old[i, j+1] - u_old[i, j-1]) / (2*dy)
                )

                v[i, j] = v_old[i, j] + dt * (
                    nu * ((v_old[i+1, j] - 2*v_old[i, j] + v_old[i-1, j]) / dx**2 + (v_old[i, j+1] - 2*v_old[i, j] + v_old[i, j-1]) / dy**2)
                    - u_old[i, j] * (v_old[i+1, j] - v_old[i-1, j]) / (2*dx)
                    - v_old[i, j] * (v_old[i, j+1] - v_old[i, j-1]) / (2*dy)
                )

        # Boundary conditions
        u[:, 0] = 1.0
        u[:, -1] = 1.0
        u[0, :] = 1.0
        u[-1, :] = 1.0

        v[:, 0] = 1.0
        v[:, -1] = 1.0
        v[0, :] = 1.0
        v[-1, :] = 1.0

    # Save the final solution
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Burgers_Equation.npy', u)
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/v_2D_Burgers_Equation.npy', v)

solve_cfd()