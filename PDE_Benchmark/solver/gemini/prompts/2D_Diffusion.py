import numpy as np

def solve_heat_equation():
    # Parameters
    nu = 1.0
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.0, 2.0
    t_final = 0.3777
    nx, ny = 50, 50  # Number of grid points in x and y
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    dt = 0.0001  # Time step size
    nt = int(t_final / dt)

    # Initialize grid
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    u = np.ones((nx, ny))

    # Initial condition
    for i in range(nx):
        for j in range(ny):
            if 0.5 <= x[i] <= 1.0 and 0.5 <= y[j] <= 1.0:
                u[i, j] = 2.0

    # Boundary conditions
    u[:, 0] = 1.0
    u[:, -1] = 1.0
    u[0, :] = 1.0
    u[-1, :] = 1.0

    # Time loop
    for n in range(nt):
        u_new = u.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_new[i, j] = u[i, j] + nu * dt * (
                    (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 +
                    (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
                )

        # Boundary conditions
        u_new[:, 0] = 1.0
        u_new[:, -1] = 1.0
        u_new[0, :] = 1.0
        u_new[-1, :] = 1.0
        u = u_new

    # Save the solution at the final time step
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Diffusion.npy', u)

solve_heat_equation()