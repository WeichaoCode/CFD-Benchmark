import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 2. # Length 
T = 1. # Time 
nx = 101 # spatial points 
nt = 100 # time points 
c = 1. # wave speed

# Discretize
dx = L / (nx - 1)
dt = T / (nt - 1)

x = np.linspace(0, L, nx)
u = np.zeros(nx)

# Initial condition
u = np.where((0.9 >= x) & (x >= 0.1), 2, 1)

for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        u[i] = un[i] - c*dt/dx*(un[i] - un[i - 1])

# Plot
plt.plot(x, u)
plt.show()