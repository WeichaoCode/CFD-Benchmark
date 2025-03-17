import numpy as np
from sympy import symbols, exp, diff
import matplotlib.pyplot as plt

nx = 101  # number of grid points
nt = 100  # number of time steps
dx = 2 * np.pi / (nx - 1)  # distance between each grid point
nu = 0.07  # viscosity
dt = dx * nu  # time step size

# Initial condition
x = np.linspace(0, 2 * np.pi, nx)
phi = exp(-x**2 /(4*nu)) + exp(-(x - 2*np.pi)**2 /(4*nu))
phiprime = -x/(2*nu)*exp(-x**2 /(4*nu)) + (x - 2*np.pi)/(2*nu)*exp(-(x - 2*np.pi)**2 /(4*nu))
u = -2 * nu * (phiprime / phi) + 4

# Numerically solve Burgers' equation
un = np.empty(nx)  # velocity field for next time step
for n in range(nt):  # loop over time steps
    un = u.copy()
    for i in range(1, nx - 1):  # loop over space
        u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i - 1]) + nu * dt / dx**2 *\
                (un[i + 1] - 2 * un[i] + un[i - 1])
    u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 *\
           (un[1] - 2 * un[0] + un[-2])
    u[-1] = u[0]
    
np.save('velocity.npy', u)  # save the velocity field

# Analytical solution for comparison
phi_analytical = exp(-(x - 4 * dt * nt)**2 /(4*nu*(nt*dt + 1))) + exp(-(x - 4 * dt * nt - 2*np.pi)**2 /(4*nu*(nt*dt + 1)))
phiprime_analytical = - (x - 4*dt*nt) /(2*nu*(nt*dt + 1))*exp(-(x - 4 * dt * nt)**2 /(4*nu*(nt*dt + 1))) + (x - 4*dt*nt - 2*np.pi) /(2*nu*(nt*dt + 1))*exp(-(x - 4 * dt * nt -2*np.pi)**2 /(4*nu*(nt*dt + 1)))
u_analytical = -2 * nu * (phiprime_analytical / phi_analytical) + 4

# Plotting
plt.figure(figsize=(11, 7), dpi=100)
plt.plot(x,u, marker='o', lw=2, label='Computational')
plt.plot(x, u_analytical, label='Analytical')
plt.xlim([0, 2 * np.pi])
plt.ylim([0, 10])
plt.legend()
plt.show()