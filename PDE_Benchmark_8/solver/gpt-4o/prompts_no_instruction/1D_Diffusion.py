import numpy as np

# Parameters
nu = 0.3  # diffusion coefficient
nx = 41  # number of spatial grid points
nt = 20  # number of time steps
sigma = 0.2  # CFL number

# Spatial domain
x = np.linspace(0, 1, nx)
dx = 2 / (nx - 1)

# Time step
dt = sigma * dx**2 / nu

# Initial condition
u = np.ones(nx)
u[int(0.5 / dx):] = 2

# Boundary conditions
u[0] = 1
u[-1] = 0

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2*un[i] + un[i-1])
    # Reapply boundary conditions
    u[0] = 1
    u[-1] = 0

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_1D_Diffusion.npy', u)