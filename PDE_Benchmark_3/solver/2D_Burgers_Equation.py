import numpy as np
import matplotlib.pyplot as plt

def solve_2d_burgers_equation(final_time, grid_points, nu):
    # Step 1: Define parameters
    dt = final_time / grid_points # time step
    dx = dy = 2 * np.pi / (grid_points - 1) # grid resolution
    x = y = np.linspace(0, 2 * np.pi, grid_points) # domain

    u = np.zeros((grid_points, grid_points)) # u velocity
    v = np.zeros((grid_points, grid_points)) # v velocity

    # MMS solution
    u_sol = lambda x, y, t: np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)
    v_sol = lambda x, y, t: np.exp(-t) * np.cos(np.pi * x) * np.cos(np.pi * y)

    # Source term from MMS
    f_u = lambda x, y, t: -u_sol(x, y, t) + nu * (np.pi**2 * u_sol(x, y, t)) + u_sol(x, y, t)**2 - v_sol(x, y, t)**2
    f_v = lambda x, y, t: -v_sol(x, y, t) + nu * (np.pi**2 * v_sol(x, y, t)) + 2 * u_sol(x, y, t) * v_sol(x, y, t)

    # Step 2: CFL condition
    CFL = lambda dt, dx, dy, u, v, nu: np.all(dt * np.abs(u) <= dx) and np.all(dt * np.abs(v) <= dy) and np.all(dt * nu <= dx * dy)
    if not CFL(dt, dx, dy, u, v, nu):
        raise Exception("CFL condition not satisfied")

    # Step 3: Compute source term from MMS solution
    source_u = f_u(x[:, None], y[None, :], 0)
    source_v = f_v(x[:, None], y[None, :], 0)

    # Step 4: Initial and boundary conditions from MMS
    u = u_sol(x[:, None], y[None, :], 0)
    v = v_sol(x[:, None], y[None, :], 0)

    X, Y = np.meshgrid(x, y)
    time = 0

    # Step 5: Solve PDE using finite differences
    while time < final_time:
        un = u.copy()
        vn = v.copy()

        u[1:-1, 1:-1] = un[1:-1, 1:-1] - dt * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1]) / dx - dt * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]) / dy + dt * nu * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dx**2 + dt * nu * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) / dy**2 + dt * source_u[1:-1, 1:-1]
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - dt * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) / dx - dt * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) / dy + dt * nu * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]) / dx**2 + dt * nu * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) / dy**2 + dt * source_v[1:-1, 1:-1]

        # Boundaries: the solution is periodic
        u[0, :] = u[-1, :]
        u[:, 0] = u[:, -1]
        u[-1, :] = u[1, :]
        u[:, -1] = u[:, 1]

        v[0, :] = v[-1, :]
        v[:, 0] = v[:, -1]
        v[-1, :] = v[1, :]
        v[:, -1] = v[:, 1]
        
        time += dt

    # Step 7: Plot numerical, exact solution and error
    u_exact = u_sol(X, Y, time) # Exact solution
    v_exact = v_sol(X, Y, time)

    err_u = np.abs(u - u_exact)
    err_v = np.abs(v - v_exact)

    plt.figure(figsize=(11, 7))
    plt.contourf(X, Y, u)
    plt.colorbar()
    plt.title('Numerical solution for u')
    plt.show()

    plt.figure(figsize=(11, 7))
    plt.contourf(X, Y, u_exact)
    plt.colorbar()
    plt.title('Exact solution for u')
    plt.show()

    plt.figure(figsize=(11, 7))
    plt.contourf(X, Y, err_u)
    plt.colorbar()
    plt.title('Error in u')
    plt.show()

    return u, v, X, Y, err_u, err_v

# Run the solver
u, v, X, Y, err_u, err_v = solve_2d_burgers_equation(final_time=10, grid_points=101, nu=0.1)