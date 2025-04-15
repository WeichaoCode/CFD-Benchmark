import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 2.0
T = 1.0
nx = 101
nt = 101
dx = L / (nx - 1)
dt = T / (nt - 1)
u_initial = 1.0
u_max = 2.0
s_loc = 0.5

# Discretize space and time
x = np.linspace(0, L, nx)
u = np.ones(nx)                     # u at time n
u_new = np.ones(nx)                 # u at time n+1
u[int(s_loc / dx):] = u_max

# Set initial condition
u[0] = u_initial

# Iterate using finite difference scheme
for n in range(nt):
    # Necessary to avoid the running off the end of the array
    for i in range(1, nx):
        CFL = u[i] * dt / dx
        u_new[i] = u[i] - CFL * (u[i] - u[i-1])
        
    u = u_new

# Plot results
plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('u')
plt.show()