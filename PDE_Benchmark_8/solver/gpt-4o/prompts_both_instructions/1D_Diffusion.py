import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.3  # diffusion coefficient
nx = 41  # number of spatial grid points
nt = 20  # number of time steps
sigma = 0.2  # CFL-like condition
dx = 2 / (nx - 1)  # spatial resolution
dt = sigma * dx**2 / nu  # time step size

# Discretized spatial domain
x = np.linspace(0, 1, nx)

# Initial condition
u = np.ones(nx)
u[int(0.5 / dx):] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2*un[i] + un[i-1])
    
    # Apply Dirichlet boundary conditions
    u[0] = 1
    u[-1] = 0

# Plot the final solution
plt.plot(x, u, label='Final Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Diffusion Equation')
plt.legend()
plt.show()

# Save the final solution to a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_1D_Diffusion.npy', u)