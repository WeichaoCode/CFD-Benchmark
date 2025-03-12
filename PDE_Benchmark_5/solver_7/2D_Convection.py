import numpy as np
import matplotlib.pyplot as plt

def initial_velocity_field(n, m):
    u = np.ones((n, m))
    v = np.ones((n, m))
    u[int(0.5*n):int(1*n), int(0.5*m):int(1*m)] = 2
    v[int(0.5*n):int(1*n), int(0.5*m):int(1*m)] = 2
    return u, v

def convection_2d(u, v, nt, dt, dx, dy):
    u_next = u.copy()
    v_next = v.copy()
    for n in range(nt):
        u_next[1:,1:] = u[1:,1:] - (u[1:,1:] * dt / dx * (u[1:,1:] - u[:-1,1:]) + v[1:,1:] * dt / dy * (u[1:,1:] - u[1:,:-1]))
        v_next[1:,1:] = v[1:,1:] - (u[1:,1:] * dt / dx * (v[1:,1:] - v[:-1,1:]) + v[1:,1:] * dt / dy * (v[1:,1:] - v[1:,:-1]))
        u_next[0, :] = 1
        u_next[-1, :] = 1
        u_next[:, 0] = 1
        u_next[:, -1] = 1
        v_next[0, :] = 1
        v_next[-1, :] = 1
        v_next[:, 0] = 1
        v_next[:, -1] = 1
        u = u_next.copy()
        v = v_next.copy()
    return u, v

if __name__ == '__main__':
    # Domain size and physical variables
    Lx = 2.0  # domain length x
    Ly = 2.0  # domain length y
    T = 0.625  # time period

    # Numerical parameters
    nx = 81  # number of x-grid points
    ny = 81  # number of y-grid points
    nt = 100  # number of time steps
    dx = Lx / (nx - 1)  # spatial step x
    dy = Ly / (ny - 1)  # spatial step y
    cfl = 0.2  # Courant number
    dt = cfl * dx  # time step

    # Initial conditions
    u, v = initial_velocity_field(nx, ny)

    # Simulation
    u, v = convection_2d(u, v, nt, dt, dx, dy)

    # Visualization
    X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
    plt.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])
    plt.show()