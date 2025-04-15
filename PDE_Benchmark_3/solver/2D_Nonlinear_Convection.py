import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y, t):
    return -np.exp(-t)* np.pi* (np.pi* np.sin(np.pi*x)* np.sin(np.pi*y)* (np.sin(np.pi*x)* np.sin(np.pi*y) + np.cos(np.pi*x)* np.cos(np.pi*y)))

def f_v(x, y, t):
    return np.exp(-t)* np.pi* (np.pi* np.cos(np.pi*x)* np.cos(np.pi*y)* (np.sin(np.pi*x)* np.sin(np.pi*y) - np.cos(np.pi*x)* np.cos(np.pi*y)))

def solve_2d_nonlinear_convection(nx, ny, nt, T, Lx, Ly):
    dt = T / (nt - 1)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    x = np.arange(0, Lx + dx, dx)
    y = np.arange(0, Ly + dy, dx)
    t = np.arange(0, T + dt, dt)

    # Initialise grid
    u, v = np.zeros((nx, ny)) , np.zeros((nx, ny))

    # Initial conditions from MMS
    u = np.exp(-0) * np.sin(np.pi*x)[:,None] * np.sin(np.pi*y)[None,:]
    v = np.exp(-0) * np.cos(np.pi*x)[:,None] * np.cos(np.pi*y)[None,:]

    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        # Compute FDM solution
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u[i, j] = un[i, j] + dt * (- un[i,j] * (un[i,j] - un[i-1,j]) / dx - vn[i,j] * (un[i,j] - un[i,j-1]) / dy + f(x[i], y[j], t[n]))
                v[i, j] = vn[i, j] + dt * (- un[i,j] * (vn[i,j] - vn[i-1,j]) / dx - vn[i,j] * (vn[i,j] - vn[i,j-1]) / dy + f_v(x[i], y[j], t[n]))

        # Enforce boundary conditions
        u[0,:] = np.exp(-t[n]) * np.sin(np.pi*x[0]) * np.sin(np.pi*y)
        u[-1,:] = np.exp(-t[n]) * np.sin(np.pi*x[-1]) * np.sin(np.pi*y)
        u[:,0] = np.exp(-t[n]) * np.sin(np.pi*x) * np.sin(np.pi*y[0])
        u[:,-1] = np.exp(-t[n]) * np.sin(np.pi*x) * np.sin(np.pi*y[-1])

        v[0,:] = np.exp(-t[n]) * np.cos(np.pi*x[0]) * np.cos(np.pi*y)
        v[-1,:] = np.exp(-t[n]) * np.cos(np.pi*x[-1]) * np.cos(np.pi*y)
        v[:,0] = np.exp(-t[n]) * np.cos(np.pi*x) * np.cos(np.pi*y[0])
        v[:,-1] = np.exp(-t[n]) * np.cos(np.pi*x) * np.cos(np.pi*y[-1])

    return u, v

# Set Parameters
nx, ny = 51, 51  # number of grid points
nt = 51  # number of time steps
T = 0.5  # final time
Lx, Ly = 1.0, 1.0  # domain size

u, v = solve_2d_non_linear_convection(nx, ny, nt, T, Lx, Ly)

# error analysis
exact_u = np.exp(-T) * np.sin(np.pi*np.linspace(0,Lx,nx))[:,None] * np.sin(np.pi*np.linspace(0,Ly,ny)) 
error_u = exact_u - u