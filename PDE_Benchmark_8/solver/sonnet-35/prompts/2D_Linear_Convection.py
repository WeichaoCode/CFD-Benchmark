import numpy as np
import matplotlib.pyplot as plt

# Problem Parameters
Lx, Ly = 2.0, 2.0  # Domain size
nx, ny = 100, 100  # Grid points
nt = 500  # Time steps
c = 1.0  # Convection speed
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = 0.001  # Time step

# Grid Generation
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initial Condition
u = np.ones((ny, nx))
mask = ((0.5 <= X) & (X <= 1.0) & (0.5 <= Y) & (Y <= 1.0))
u[mask] = 2.0

# Boundary Conditions
u[0, :] = 1.0   # Bottom boundary
u[-1, :] = 1.0  # Top boundary
u[:, 0] = 1.0   # Left boundary
u[:, -1] = 1.0  # Right boundary

# Lax-Wendroff Method
def lax_wendroff_2d(u, c, dt, dx, dy):
    ny, nx = u.shape
    u_new = np.zeros_like(u)
    
    # Interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # x-direction flux
            fx_p = 0.5 * c * (u[i,j+1] + u[i,j]) - 0.5 * (c**2 * dt/dx) * \
                   (u[i,j+1] - u[i,j])
            fx_m = 0.5 * c * (u[i,j] + u[i,j-1]) - 0.5 * (c**2 * dt/dx) * \
                   (u[i,j] - u[i,j-1])
            
            # y-direction flux
            fy_p = 0.5 * c * (u[i+1,j] + u[i,j]) - 0.5 * (c**2 * dt/dy) * \
                   (u[i+1,j] - u[i,j])
            fy_m = 0.5 * c * (u[i,j] + u[i-1,j]) - 0.5 * (c**2 * dt/dy) * \
                   (u[i,j] - u[i-1,j])
            
            # Update
            u_new[i,j] = u[i,j] - (dt/dx) * (fx_p - fx_m) - (dt/dy) * (fy_p - fy_m)
    
    # Enforce boundary conditions
    u_new[0, :] = 1.0
    u_new[-1, :] = 1.0
    u_new[:, 0] = 1.0
    u_new[:, -1] = 1.0
    
    return u_new

# Time integration
for _ in range(nt):
    u = lax_wendroff_2d(u, c, dt, dx, dy)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/u_2D_Linear_Convection.npy', u)