import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Nx = 41  
Lx = 2.0  
dx = Lx / (Nx - 1)  
dt = 0.0125  
Nt = 20  

# Create spatial grid
x = np.linspace(0, Lx, Nx)

# Initialize u
u = np.ones(Nx)
u[(x >= 0.5) & (x <= 1)] = 2  

# Time integration using Upwind Scheme
for n in range(Nt):
    u_new = u.copy()
    for i in range(1, Nx):
        u_new[i] = u[i] - u[i] * dt / dx * (u[i] - u[i-1])
    u = u_new.copy()

# Plot result
plt.plot(x, u)
plt.show()