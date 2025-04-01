import numpy as np

# Problem parameters
nx = 41
nt = 20
nu = 0.3
sigma = 0.2
L = 2.0

# Grid setup
dx = L / (nx - 1)
dt = sigma * dx**2 / nu

# Initialize solution array
x = np.linspace(0, L, nx)
u = np.zeros(nx)

# Initial condition
u[(x >= 0.5) & (x <= 1)] = 2
u[(x < 0.5) | (x > 1)] = 1

# Time marching
for _ in range(nt):
    u_old = u.copy()
    
    # Finite difference discretization
    u[1:-1] = u_old[1:-1] + nu * dt/dx**2 * (u_old[2:] - 2*u_old[1:-1] + u_old[:-2])
    
    # Neumann boundary conditions (zero flux)
    u[0] = u[1]
    u[-1] = u[-2]

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_1D_Diffusion.npy', u)