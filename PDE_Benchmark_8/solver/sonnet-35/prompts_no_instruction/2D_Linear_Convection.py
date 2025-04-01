import numpy as np

# Domain parameters
nx, ny = 81, 81
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Convection speed
c = 1.0

# Time parameters
nt = 100
sigma = 0.2
dt = sigma * min(dx, dy) / c

# Initialize solution array
u = np.ones((ny, nx))

# Initial condition
mask_x = (x >= 0.5) & (x <= 1.0)
mask_y = (y >= 0.5) & (y <= 1.0)
X, Y = np.meshgrid(x, y)
u[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2.0

# Time-stepping using finite difference method
for _ in range(nt):
    u_old = u.copy()
    
    # Update interior points using central differencing
    u[1:-1, 1:-1] = u_old[1:-1, 1:-1] - c * dt/dx * (u_old[1:-1, 1:-1] - u_old[1:-1, 0:-2]) \
                                       - c * dt/dy * (u_old[1:-1, 1:-1] - u_old[0:-2, 1:-1])

    # Enforce boundary conditions
    u[0, :] = 1.0   # Bottom boundary
    u[-1, :] = 1.0  # Top boundary
    u[:, 0] = 1.0   # Left boundary
    u[:, -1] = 1.0  # Right boundary

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_2D_Linear_Convection.npy', u)