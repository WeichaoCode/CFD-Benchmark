import numpy as np

# Problem parameters
Lx, Ly = 2.0, 2.0  # Domain size
nx, ny = 100, 100  # Grid points
nu = 0.05  # Diffusion coefficient
t_start, t_end = 0.0, 0.3777  # Time domain

# Grid generation
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Time stepping
dt = 0.001  # Time step
nt = int((t_end - t_start) / dt)

# Initialize solution array
u = np.ones((ny, nx))

# Initial condition
mask = (x >= 0.5) & (x <= 1.0)
mask2d = np.outer(mask, mask)
u[mask2d] = 2.0

# Finite difference solver
def diffusion_2d(u, nu, dx, dy, dt):
    u_new = u.copy()
    
    # Interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            u_new[i,j] = u[i,j] + nu * dt * (
                (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx**2 +
                (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy**2
            )
    
    # Boundary conditions
    u_new[0,:] = 1.0  # Bottom boundary
    u_new[-1,:] = 1.0  # Top boundary
    u_new[:,0] = 1.0  # Left boundary
    u_new[:,-1] = 1.0  # Right boundary
    
    return u_new

# Time marching
for _ in range(nt):
    u = diffusion_2d(u, nu, dx, dy, dt)

# Save final solution
np.save('/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_2D_Diffusion.npy', u)