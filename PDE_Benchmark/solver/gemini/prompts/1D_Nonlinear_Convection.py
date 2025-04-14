import numpy as np

def solve_burgers():
    # Parameters
    nx = 100
    nt = 500
    dx = 2 * np.pi / nx
    dt = 0.01
    t_final = 5.0

    # Domain
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)

    # Initial condition
    u = np.sin(x) + 0.5 * np.sin(0.5 * x)

    # Numerical solution (Lax-Friedrichs)
    for n in range(nt):
        u_old = u.copy()
        for i in range(nx):
            u[i] = 0.5 * (u_old[(i - 1) % nx] + u_old[(i + 1) % nx]) - \
                   0.5 * dt / dx * (u_old[(i + 1) % nx]**2 - u_old[(i - 1) % nx]**2) / 2.0

        time = (n + 1) * dt
        if time >= t_final:
            break

    # Save the solution at the final time step
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_1D_Nonlinear_Convection.npy', u)

solve_burgers()