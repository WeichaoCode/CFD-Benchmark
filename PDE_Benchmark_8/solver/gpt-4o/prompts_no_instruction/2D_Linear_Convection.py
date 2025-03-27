import numpy as np

# Parameters
nx, ny = 81, 81
nt = 100
c = 1.0
sigma = 0.2
dx = dy = 2.0 / (nx - 1)
dt = sigma * min(dx, dy) / c

# Initialize the solution array
u = np.ones((ny, nx))

# Initial conditions
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    u[1:, 1:] = (un[1:, 1:] - 
                 c * dt / dx * (un[1:, 1:] - un[1:, :-1]) - 
                 c * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_2D_Linear_Convection.npy', u)