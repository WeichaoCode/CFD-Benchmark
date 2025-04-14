import numpy as np

def solve_pde():
    # Parameters
    c = 1.0
    epsilon = 5e-4
    x_start = -5.0
    x_end = 5.0
    t_start = 0.0
    t_end = 10.0
    nx = 100
    nt = 500
    dx = (x_end - x_start) / (nx - 1)
    dt = (t_end - t_start) / (nt - 1)

    # Initialize grid
    x = np.linspace(x_start, x_end, nx)
    u = np.zeros(nx)

    # Initial condition
    u[:] = np.exp(-x[:]**2)

    # Time loop
    for n in range(nt - 1):
        u_new = np.zeros(nx)

        # Spatial loop with periodic boundary conditions
        for i in range(nx):
            i_minus = (i - 1) % nx
            i_plus = (i + 1) % nx

            # FTCS scheme
            u_new[i] = u[i] - c * dt / (2 * dx) * (u[i_plus] - u[i_minus]) + \
                       epsilon * dt / dx**2 * (u[i_plus] - 2 * u[i] + u[i_minus])

        u[:] = u_new[:]

    # Save the final solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_1D_Linear_Convection.npy', u)

solve_pde()