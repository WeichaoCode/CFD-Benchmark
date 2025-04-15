import numpy as np
import matplotlib.pyplot as plt

# Variable Declarations
nx = 41
nt = 20
nu = 0.3
sigma = 0.2
dx = 2 / (nx - 1)
dt = sigma * dx**2 / nu

# Initialization
u = np.ones(nx)
u[int(0.5 / dx):int(1 / dx + 1)] = 2

un = np.ones(nx)

# Numerical Solution
for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])

# Boundary Conditions
u[0] = 1
u[-1] = 0

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/u_1D_Diffusion.npy', u)

# Visualization
plt.plot(np.linspace(0, 2, nx), u)
plt.show()