import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 1.0   # length of the domain
T = 1.0   # time of the simulation
nx = 101  # number of spatial points in the grid
nt = 101  # number of time steps
c = 1.0   # wave speed

dx = L / (nx - 1)  # spatial discretization size
dt = T / (nt - 1)  # initial temporal discretization

# Adjust dt to ensure CFL condition is met
dt = 0.8 * dx / c  # 0.8 is used as a safety factor

# Initial condition
u0 = np.where((0.25 < np.arange(nx)*dx) & (np.arange(nx)*dx < 0.5), 1, 0)

# Time integration (Explicit Euler)
u = np.copy(u0)
for n in range(nt):
    u[1:nx-1] = u[1:nx-1] - c * dt / (2*dx) * (u[2:nx] - u[:nx-2])

    if n % 20 == 0:  # Plot every 20 timesteps
        plt.figure(figsize=(6, 4), dpi=80)
        plt.plot(np.linspace(0, L, nx), u, label='t = {0:.2f}'.format(n*dt))
        plt.legend(loc='upper left')
        plt.ylim(0, 1.1)
        plt.xlabel('x')
        plt.ylabel('u')
        plt.grid(True)
        plt.title('1D Linear Convection')

plt.show()