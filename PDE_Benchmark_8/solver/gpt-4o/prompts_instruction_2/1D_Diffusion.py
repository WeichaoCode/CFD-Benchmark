import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 41
nt = 20
nu = 0.3
sigma = 0.2

# Spatial and temporal discretization
dx = 2 / (nx - 1)
dt = sigma * dx**2 / nu

# Initialize the solution array
u = np.ones(nx)
x = np.linspace(0, 1, nx)

# Apply initial conditions
u[int(0.5 / dx):] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2*un[i] + un[i-1])
    
    # Apply Dirichlet boundary conditions
    u[0] = 1
    u[-1] = 0

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/u_1D_Diffusion.npy', u)

# Plot the final solution
plt.plot(x, u, label='Final Solution')
plt.xlabel('Spatial coordinate x')
plt.ylabel('u(x, t)')
plt.title('1D Diffusion Equation Solution at Final Time Step')
plt.legend()
plt.show()