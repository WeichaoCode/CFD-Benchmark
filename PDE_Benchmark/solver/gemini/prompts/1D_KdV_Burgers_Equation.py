import numpy as np

def solve_kdv_burgers():
    # Parameters
    L = 10.0
    T = 10.0
    nx = 200
    nt = 500
    a = 1e-4
    b = 2e-4
    n = 20

    # Grid
    dx = L / nx
    dt = T / nt
    x = np.linspace(0, L, nx, endpoint=False)

    # Initial condition
    u = 0.5 / n * np.log(1 + np.cosh(n)**2 / np.cosh(n * (x - 0.2 * L))**2)

    # Numerical method (FTCS for advection, central difference for diffusion and dispersion)
    for _ in range(nt):
        u_new = np.copy(u)
        for i in range(nx):
            u_new[i] = u[i] - dt * u[i] * (u[(i+1)%nx] - u[(i-1)%nx]) / (2*dx) + \
                       a * dt * (u[(i+1)%nx] - 2*u[i] + u[(i-1)%nx]) / dx**2 + \
                       b * dt * (u[(i+2)%nx] - 2*u[(i+1)%nx] + 2*u[(i-1)%nx] - u[(i-2)%nx]) / (2*dx**3)
        u = u_new

    # Save the solution at the final time step
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_1D_KdV_Burgers_Equation.npy', u)

solve_kdv_burgers()