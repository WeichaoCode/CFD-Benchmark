import numpy as np

# Parameters
nu = 0.3  # diffusion coefficient
L = 2.0   # domain length
T = 0.0333  # total time

# Numerical parameters
nx = 200  # number of spatial points
dx = L/nx
dt = 0.5 * dx**2 / nu  # stability condition
nt = int(T/dt)  # number of time steps

# Initialize grid and solution
x = np.linspace(0, L, nx)
u = np.ones(nx)  # initialize with u=1 everywhere

# Set initial condition
u[(x >= 0.5) & (x <= 1.0)] = 2.0

# Time stepping
for n in range(nt):
    un = u.copy()
    
    # Interior points
    u[1:-1] = un[1:-1] + nu*dt/dx**2 * (un[2:] - 2*un[1:-1] + un[:-2])
    
    # Neumann boundary conditions
    u[0] = u[1]  # du/dx = 0 at x=0
    u[-1] = u[-2]  # du/dx = 0 at x=L

# Save final solution
np.save('u', u)