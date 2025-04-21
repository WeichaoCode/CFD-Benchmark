import numpy as np

def solve_shock_tube():
    # Parameters
    gamma = 1.4
    nx = 200
    nt = 500
    xmin = -1.0
    xmax = 1.0
    t_final = 0.25

    # Grid
    x = np.linspace(xmin, xmax, nx)
    dx = x[1] - x[0]
    dt = t_final / nt

    # Initial conditions
    rho = np.where(x < 0, 1.0, 0.125)
    u = np.zeros_like(x)
    p = np.where(x < 0, 1.0, 0.1)
    E = p / ((gamma - 1) * rho) + 0.5 * u**2

    # Conservative variables
    U = np.array([rho, rho * u, rho * E])

    # Numerical flux (Lax-Friedrichs)
    def flux(U):
        rho = U[0]
        v = U[1] / rho
        E = U[2] / rho
        p = (gamma - 1) * rho * (E - 0.5 * v**2)
        F = np.array([rho * v, rho * v**2 + p, v * (rho * E + p)])
        return F

    def max_eigenvalue(U):
        rho = U[0]
        v = U[1] / rho
        E = U[2] / rho
        p = (gamma - 1) * rho * (E - 0.5 * v**2)
        a = np.sqrt(gamma * p / rho)
        return np.max(np.abs(v) + a)

    # Time loop
    for n in range(nt):
        F = flux(U)
        max_speed = max_eigenvalue(U)
        dt = 0.5 * dx / max_speed

        F_left = np.zeros_like(F)
        F_right = np.zeros_like(F)

        for i in range(1, nx - 1):
            F_left[:, i] = 0.5 * (F[:, i] + F[:, i - 1]) - 0.5 * dx / dt * (U[:, i] - U[:, i - 1])
            F_right[:, i] = 0.5 * (F[:, i + 1] + F[:, i]) - 0.5 * dx / dt * (U[:, i + 1] - U[:, i])

        # Boundary conditions (reflective)
        F_left[:, 0] = F_left[:, 1]
        F_right[:, nx - 1] = F_right[:, nx - 2]

        # Update
        for i in range(1, nx - 1):
            U[:, i] = U[:, i] - dt / dx * (F_right[:, i] - F_left[:, i])

        # Update primitive variables
        rho = U[0]
        u = U[1] / rho
        E = U[2] / rho
        p = (gamma - 1) * rho * (E - 0.5 * u**2)

    # Save the final solution
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/rho_1D_Euler_Shock_Tube.npy', rho)
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/u_1D_Euler_Shock_Tube.npy', u)
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/p_1D_Euler_Shock_Tube.npy', p)

solve_shock_tube()