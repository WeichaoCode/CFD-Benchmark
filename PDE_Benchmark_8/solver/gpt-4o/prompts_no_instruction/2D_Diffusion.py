import numpy as np

# Parameters
nu = 1.0  # diffusion coefficient
nx = ny = 31  # number of grid points
dx = dy = 2.0 / (nx - 1)  # grid spacing
sigma = 0.25  # CFL number
dt = sigma * dx * dy / nu  # time step size
nt = 50  # number of time steps

# Initialize the solution array
u = np.ones((ny, nx))

# Set initial conditions
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Apply initial condition: u = 2 in the region 0.5 <= x, y <= 1
u[(X >= 0.5) & (X <= 1) & (Y >= 0.5) & (Y <= 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    # Update the solution using finite difference method
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                     nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                     nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_2D_Diffusion.npy', u)