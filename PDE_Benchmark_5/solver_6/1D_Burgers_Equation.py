import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 2 * np.pi  # length of domain
T = 1.0  # time of simulation
ν = 0.07  # viscosity
nx = 101  # number of spatial grids
nt = 100  # number of time steps
dx = L / (nx - 1)  # spatial grid size
dt = T / nt  # time step size

# Ensure Stability through CFL condition
CFL = ν * dt / dx**2 
if CFL >= 0.5: 
    print("Warning: CFL condition not met")

# Space discretization
x = np.linspace(0, L, nx)

# Initial condition
φ = np.exp(-x**2 / (4 * ν)) + np.exp(-(x - L)**2 / (4 * ν))
u = -2 * ν * (np.exp(-x**2 / (4 * ν)) - np.exp(-(x - L)**2 / (4 * ν))) / φ + 4 

# Time integration
for n in range(nt):  
    un = u.copy()
    u[1:-1] = un[1:-1] - un[1:-1] * dt / dx * (un[1:-1] - un[:-2]) + ν * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2]) 
    # Periodic BC
    u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + ν * dt / dx**2 * (un[1] - 2 * un[0] + un[-2]) 
    u[-1] = u[0]

# Analytic solution
φ = np.exp(-(x - 4 * T)**2 / (4 * ν * (T + 1))) + np.exp(-(x - 4 * T - 2 * np.pi)**2 / (4 * ν * (T + 1)))
u_exact = -2 * ν * (np.exp(-(x - 4 * T)**2 / (4 * ν * (T + 1))) - np.exp(-(x - 4 * T - 2 * np.pi)**2 / (4 * ν * (T + 1)))) / φ + 4 
   
# Plot solution
plt.figure(figsize=(11, 7), dpi=100)
plt.plot(x, u, '-bo', label='Numerical')
plt.plot(x, u_exact, 'k', label='Analytical')
plt.xlim([0, 2 * np.pi])
plt.ylim([0, 10])
plt.legend()
plt.show()