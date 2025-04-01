import numpy as np

# Parameters
nu = 0.3
nx = 41
nt = 20
sigma = 0.2
dx = 2 / (nx - 1)
dt = sigma * dx**2 / nu

# Spatial grid
x = np.linspace(0, 2, nx)

# Initial condition
u = np.where((x >= 0.5) & (x <= 1.0), 2.0, 1.0)

# Time-stepping
for _ in range(nt):
    un = u.copy()
    u[1:-1] = un[1:-1] + nu * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2])
    # Boundary conditions: u=1 at both ends
    u[0] = 1.0
    u[-1] = 1.0

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/u_1D_Diffusion.npy', u)