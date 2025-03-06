import numpy as np
import matplotlib.pyplot as plt


def f(x, t):
    """Source term computed from Manufactured Solution."""
    return np.exp(-t) * np.sin(np.pi * x) * (1 - np.pi)


def u_exact(x, t):
    """Manufactured Solution."""
    return np.exp(-t) * np.sin(np.pi * x)


def solve_1d_nonlinear_convection(nx=100, nt=200, T=1.0, L=1.0):
    """Solve 1D Nonlinear Convection PDE with Manufactured Solution."""

    # Step 1: Define parameters
    dx = L / (nx - 1)  # grid size
    dt = T / nt  # time step
    x = np.linspace(0, L, nx)  # grid points in space
    t = np.linspace(0, T, nt)  # grid points in time

    # Step 2: Check CFL condition (Courant–Friedrichs–Lewy condition)
    assert dt <= dx ** 2, "Instability due to CFL condition violated"

    # Step 3: Compute source term
    f_vec = f(x, t=None)

    # Step 4: Compute initial and boundary conditions
    u = np.empty((nt, nx))
    u[0, :] = u_exact(x, 0)
    u[:, 0] = u_exact(0, t)
    u[:, -1] = u_exact(L, t)

    # Step 5: Solve the PDE using finite difference
    for n in range(nt - 1):  # time loop
        for i in range(1, nx - 1):  # space loop
            u[n + 1, i] = u[n, i] - u[n, i] * (dt / dx) * (u[n, i] - u[n, i - 1]) + dt * f_vec[i]

    # Step 6: Compute exact solution for comparison
    u_ex = u_exact(x, T)

    # Step 7: Error analysis and plot numerical, exact solution and error
    error = np.abs(u[-1, :] - u_ex)
    plt.figure(figsize=(9, 6))
    plt.plot(x, u[-1, :], label='Numerical')
    plt.plot(x, u_ex, label='Exact')
    plt.plot(x, error, label='Error')
    plt.legend()
    plt.title('Numerical & Exact solutions and Error at T={}'.format(T))
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid(True)
    plt.show()

    return u


# use the function
u = solve_1d_nonlinear_convection()
