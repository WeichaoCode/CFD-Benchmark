import numpy as np
import matplotlib.pyplot as plt

# 1. Define parameters
L = 2*np.pi   # length of spatial domain
T = 0.5   # time of solution
nx = 101  # number of spatial points in grid
nt = 100  # number of time steps
nu = 0.07 # viscosity

dx = L / (nx - 1)  # spatial discretization size
dt = T / (nt - 1)  # time discretization size

# 2. Discretize space and time
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# 3. Initialize the velocity field
u = np.zeros((nt, nx))
u[0,:] = -2 * nu * (np.exp(-x**2/(4*nu)) + np.exp(-(x-L)**2/(4*nu))) / (np.exp(-x**2/(4*nu)) + np.exp(-(x-L)**2/(4*nu))) + 4

# 4. Iterate using the finite difference scheme
for n in range(nt-1):
    u[n+1, 1:-1] = (u[n, 1:-1] -
                    u[n, 1:-1] * dt / dx * (u[n, 1:-1] - u[n, :-2]) +
                    nu * dt / dx**2 * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]))
                    
    # 5. Apply periodic boundary conditions
    u[n+1, 0] = u[n+1, -2]
    u[n+1, -1] = u[n+1, 1]

# 7. Plot and visualize the solution
for n in range(nt):
    if n % 20 == 0:  # plot every 20th time step
        plt.plot(x,u[n,:])
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Burgers’ equation - FDM solution')
plt.show()