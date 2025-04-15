import numpy as np
import matplotlib.pyplot as plt

# Define the Manufactured solution
def u_exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y)

def v_exact(x, y, t):
    return np.exp(-t) * np.cos(np.pi*x) * np.cos(np.pi*y)

# Define the source term by substituting the exact solution into the PDE.
def f_exact(x, y, t):
    u = u_exact(x, y, t)
    v = v_exact(x, y, t)
    return -u + np.pi * (2*u - v*np.sin(np.pi*x) - u*np.sin(np.pi*y)) * np.exp(-t)

def fv_exact(x, y, t):
    u = u_exact(x, y, t)
    v = v_exact(x, y, t)
    return -v + np.pi * (2*v - u*np.cos(np.pi*x) - v*np.cos(np.pi*y)) * np.exp(-t)

# Define the 2D nonlinear convection model
def convection2D(nx, ny, nt, dt, dx, dy):
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    f = np.zeros((ny, nx))
    f_v = np.zeros((ny, nx))

    # Initialize with exact solutions at t=0
    for i in range(nx):
        for j in range(ny):
                u[j,i] = u_exact(i*dx, j*dy, 0)
                v[j,i] = v_exact(i*dx, j*dy, 0)

    # Time stepping
    for t in range(nt):
        un = u.copy()
        vn = v.copy()
        # Update source terms
        for i in range(nx):
            for j in range(ny):
                f[j,i] = f_exact(i*dx, j*dy, t*dt)
                f_v[j,i] = fv_exact(i*dx, j*dy, t*dt)

        # Finite difference scheme (explicit method with upwind differencing for convective terms)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u[j, i] = (un[j,i] - un[j,i]*dt/dx*(un[j,i]-un[j,i-1]) - vn[j,i]*dt/dy*(un[j,i]-un[j-1,i])) + dt*f[j,i]
                v[j, i] = (vn[j,i] - un[j,i]*dt/dx*(vn[j,i]-vn[j,i-1]) - vn[j,i]*dt/dy*(vn[j,i]-vn[j-1,i])) + dt*f_v[j,i]

    return u, v

# Tests
nx = 101
ny = 101
nt = 100
dx = 1 / (nx - 1)
dy = 1 / (ny - 1)
dt = 0.01

u_comp, v_comp = convection2D(nx, ny, nt, dt, dx, dy)

fig = plt.figure(figsize=(11, 7), dpi=100)
plt.contourf(u_comp, cmap='viridis')
plt.colorbar()

# Compute exact solution for comparison
u_exact_sol = np.zeros((ny, nx))
v_exact_sol = np.zeros((ny, nx))

for i in range(nx):
    for j in range(ny):
            u_exact_sol[j,i] = u_exact(i*dx, j*dy, nt*dt)
            v_exact_sol[j,i] = v_exact(i*dx, j*dy, nt*dt)

# Compute absolute error
error_u = np.abs(u_exact_sol - u_comp)
error_v = np.abs(v_exact_sol - v_comp)

# Plot error
fig = plt.figure(figsize=(11, 7), dpi=100)
plt.contourf(error_u, cmap='viridis')
plt.colorbar()