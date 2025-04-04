import numpy as np

# Parameters
c = 1.0
x_min, x_max = 0.0, 2.0
y_min, y_max = 0.0, 2.0
t_max = 0.5

# Discretization
nx, ny = 101, 101  # number of grid points
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)

# CFL condition
sigma = 0.5
dt = sigma * min(dx, dy) / (c * np.sqrt(2))
nt = int(np.ceil(t_max / dt))
dt = t_max / nt  # adjust dt to reach exactly t_max

# Create grid
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initial condition
u = np.ones((nx, ny))
u[np.where((X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0))] = 2.0

# Time-stepping
for n in range(nt):
    un = u.copy()
    # Compute derivatives using upwind scheme
    du_dx = (un - np.roll(un, 1, axis=0)) / dx
    du_dy = (un - np.roll(un, 1, axis=1)) / dy
    u = un - c * dt * (du_dx + du_dy)
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1.0
    u[-1, :] = 1.0
    u[:, 0] = 1.0
    u[:, -1] = 1.0

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/u_2D_Linear_Convection.npy', u)