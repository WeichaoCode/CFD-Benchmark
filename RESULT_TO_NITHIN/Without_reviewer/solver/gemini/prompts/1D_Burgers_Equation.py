import numpy as np

def solve_burgers():
    # Parameters
    nu = 0.07
    nx = 101
    nt = 100
    dx = 2 * np.pi / (nx - 1)
    dt = 0.0014 * np.pi
    x = np.linspace(0, 2 * np.pi, nx)

    # Initial condition
    phi = np.exp(-(x**2) / (4 * nu)) + np.exp(-((x - 2 * np.pi)**2) / (4 * nu))
    dphidx = (-x / (2 * nu)) * np.exp(-(x**2) / (4 * nu)) + (-(x - 2 * np.pi) / (2 * nu)) * np.exp(-((x - 2 * np.pi)**2) / (4 * nu))
    u = -2 * nu * (dphidx / phi) + 4

    # Time loop
    for n in range(nt):
        un = u.copy()
        for i in range(1, nx - 1):
            u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1]) + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])

        # Periodic boundary conditions
        u[0] = un[0] - un[0] * dt / dx * (un[0] - un[nx-2]) + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[nx-2])
        u[nx-1] = un[nx-1] - un[nx-1] * dt / dx * (un[nx-1] - un[nx-2]) + nu * dt / dx**2 * (u[1] - 2 * un[nx-1] + un[nx-2])

    # Save the final solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_1D_Burgers_Equation.npy', u)

solve_burgers()