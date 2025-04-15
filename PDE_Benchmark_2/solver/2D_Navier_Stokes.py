import numpy as np
import math

# Function defining the manufactured solutions
def mms_solution(x, y, t):
    u = math.exp(-t) * np.sin(math.pi*x) * np.sin(math.pi*y)
    v = math.exp(-t) * np.cos(math.pi*x) * np.cos(math.pi*y)
    p = math.exp(-t) * np.cos(math.pi*x) * np.cos(math.pi*y)
    return u, v, p

def solver(nt, nx, ny, dt, dx, dy, nu, rho):
    # Define the grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    u, v, p = np.zeros((nx, ny)), np.zeros((nx, ny)), np.zeros((nx, ny))
    
    # Iterate through time
    for t in range(nt):
        # Filling the update of u, v and p using the Navier Stokes equations
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                un, vn, pn = mms_solution(x[i], y[j], t*dt)
                
                u[i,j] = un + dt*(-un*(un - u[i-1,j])/dx - vn*(un - u[i,j-1])/dy + nu*((u[i+1,j] - 2*un + u[i-1,j])/(dx**2) + (u[i,j+1] - 2*un + u[i,j-1])/(dy**2)) - (pn - p[i-1,j])/dx)
                
                v[i,j] = vn + dt*(-un*(vn - v[i-1,j])/dx - vn*(vn - v[i,j-1])/dy + nu*((v[i+1,j] - 2*vn + v[i-1,j])/(dx**2) + (v[i,j+1] - 2*vn + v[i,j-1])/(dy**2)) - (pn - p[i,j-1])/dy)
                
                p[i,j] = pn + dt*((p[i+1,j] - 2*pn + p[i-1,j])/(dx**2) + (p[i,j+1] - 2*pn + p[i,j-1])/(dy**2))

        # Enforcing boundary conditions
        u[0, :], u[-1, :], u[:, 0], u[:, -1] = mms_solution(x[0], y, dt*t)
        v[0, :], v[-1, :], v[:, 0], v[:, -1] = mms_solution(x, y[0], dt*t)
        p[0, :], p[-1, :], p[:, 0], p[:, -1] = mms_solution(x, y, dt*t)

    absolute_error = np.sqrt((u - mms_solution(x, y, nt*dt)[0])**2 + (v - mms_solution(x, y, nt*dt)[1])**2 + (p - mms_solution(x, y, nt*dt)[2])**2)

    return u, v, p, absolute_error

# The solver can then be called with the appropriate parameters
nt = 100  # number of time steps
nx = 50  # number of grid points in x direction
ny = 50  # number of grid points in y direction
dt = 0.01  # time step size
dx = 1.0 / (nx - 1)  # grid size in x direction
dy = 1.0 / (ny - 1)  # grid size in y direction
nu = 0.1  # viscosity
rho = 1.0  # density

u, v, p, absolute_error = solver(nt, nx, ny, dt, dx, dy, nu, rho)