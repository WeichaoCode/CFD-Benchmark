import numpy as np

# Parameters
nu = 1.0
nx, ny = 31, 31
dx = 2.0 / (nx - 1)
dy = 2.0 / (ny - 1)
nt = 50
sigma = 0.25
dt = sigma * dx * dy / nu

# Initialize u
u = np.ones((ny, nx))
# Set initial condition: u=2 in 0.5 <= x,y <=1
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)
u[np.where((X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0))] = 2.0

# Time-stepping
for _ in range(nt):
    un = u.copy()
    u[1:-1,1:-1] = un[1:-1,1:-1] + nu * dt * (
        (un[1:-1,2:] - 2 * un[1:-1,1:-1] + un[1:-1,0:-2]) / dx**2 +
        (un[2:,1:-1] - 2 * un[1:-1,1:-1] + un[0:-2,1:-1]) / dy**2
    )
    # Apply Dirichlet boundary conditions
    u[0,:] = 1
    u[-1,:] = 1
    u[:,0] = 1
    u[:,-1] = 1

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/u_2D_Diffusion.npy', u)