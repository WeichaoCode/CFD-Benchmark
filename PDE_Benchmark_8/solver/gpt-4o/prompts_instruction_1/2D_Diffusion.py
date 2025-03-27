import numpy as np

# Parameters
nx, ny = 31, 31
Lx, Ly = 2.0, 2.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
nu = 1.0
sigma = 0.25
dt = sigma * dx * dy / nu
nt = 50

# Initialize the solution array
u = np.ones((ny, nx))

# Set initial conditions
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    # Update the solution using the finite difference scheme
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                     nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                     nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_1/u_2D_Diffusion.npy', u)