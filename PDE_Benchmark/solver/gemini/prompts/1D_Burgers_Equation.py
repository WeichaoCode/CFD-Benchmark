import numpy as np

def solve_burgers():
    # Parameters
    nx = 101  # Number of spatial points
    nt = 200  # Number of time steps
    dx = 2 * np.pi / (nx - 1)
    nu = 0.07
    dt = 0.0007  # Time step size

    x = np.linspace(0, 2 * np.pi, nx)

    # Initial condition
    phi = np.exp(-(x**2) / (4 * nu)) + np.exp(-((x - 2 * np.pi)**2) / (4 * nu))
    dphidx = (-x / (2 * nu)) * np.exp(-(x**2) / (4 * nu)) + (-(x - 2 * np.pi) / (2 * nu)) * np.exp(-((x - 2 * np.pi)**2) / (4 * nu))
    u = -2 * nu * (dphidx / phi) + 4

    # Time loop
    for n in range(nt):
        u_old = u.copy()
        for i in range(1, nx - 1):
            u[i] = u_old[i] - u_old[i] * (dt / dx) * (u_old[i] - u_old[i - 1]) + nu * (dt / dx**2) * (u_old[i + 1] - 2 * u_old[i] + u_old[i - 1])

        # Periodic boundary conditions
        u[0] = u_old[0] - u_old[0] * (dt / dx) * (u_old[0] - u_old[nx - 2]) + nu * (dt / dx**2) * (u_old[1] - 2 * u_old[0] + u_old[nx - 2])
        u[nx - 1] = u[0]

    # Save the final solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_1D_Burgers_Equation.npy', u)

solve_burgers()