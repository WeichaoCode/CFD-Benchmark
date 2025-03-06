import numpy as np
import matplotlib.pyplot as plt
from math import pi, exp, sin


def solve_1d_burgers_equation():
    # Step 1: Define PARAMETERS
    nx = 101
    L = 1.0
    dx = L / (nx - 1)
    nt = 100
    T = 0.5
    dt = T / nt
    nu = 0.07
    x = np.linspace(0.0, L, num=nx)
    u = np.zeros(nx)
    u_n = np.zeros(nx)
    f = np.zeros(nx)

    # Step 2: Check CFL CONDITION
    sigma = 0.5
    dt = sigma * dx ** 2 / nu
    nt = int(T / dt)

    # Step 3: compute source term from MMS solution
    for i in range(nx):
        f[i] = 2.0 * nu * pi ** 2 * sin(pi * x[i]) * exp(-t) - pi * cos(pi * x[i]) * exp(-t)

    # Step 4: compute the initial and boundary conditions from MMS
    for i in range(nx):
        u_n[i] = u[i] = sin(pi * x[i])

    # Step 5: solve the PDE using FINITE DIFFERENCE
    for n in range(nt):
        u_n = u.copy()
        u[1:-1] = (u_n[1:-1] -
                   dt / dx * u_n[1:-1] * (u_n[1:-1] - u_n[:-2]) +
                   nu * dt / dx ** 2 * (u_n[2:] - 2 * u_n[1:-1] + u_n[:-2]) +
                   dt * f[1:-1])

    # Step 6: compute exact solution for comparison
    u_exact = np.array([sin(pi * xi) * exp(-T) for xi in x])

    # Step 7: error analysis and plot numerical and exact solution
    error = np.sqrt(dx * np.sum((u - u_exact) ** 2))
    print('Relative L2 error:', error)
    plt.figure(figsize=(9, 6))
    plt.plot(x, u, label='Numerical')
    plt.plot(x, u_exact, label='Exact')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


solve_1d_burgers_equation()
