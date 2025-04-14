import numpy as np

# Problem parameters
Lx, Ly = 2.0, 2.0  # Domain size
nx, ny = 100, 100  # Grid points
nu = 1.0  # Diffusion coefficient
t_start, t_end = 0.0, 0.3777  # Time domain

# Grid generation
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Stability condition (CFL)
alpha = nu
dt = 0.25 * min(dx, dy)**2 / alpha  # Ensure numerical stability
nt = int((t_end - t_start) / dt)

# Initialize solution array
u = np.ones((ny, nx), dtype=np.float64)

# Initial condition
mask = ((x >= 0.5) & (x <= 1.0))[:, np.newaxis] & ((y >= 0.5) & (y <= 1.0))
u[mask] = 2.0

# Boundary conditions
u[0, :] = 1.0   # Bottom boundary
u[-1, :] = 1.0  # Top boundary
u[:, 0] = 1.0   # Left boundary
u[:, -1] = 1.0  # Right boundary

# Finite difference method (explicit)
def diffusion_2d(u, nu, dx, dy, dt):
    u_new = u.copy()
    
    # Interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # 2D diffusion equation discretization
            laplace_x = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
            laplace_y = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            u_new[i, j] = u[i, j] + nu * dt * (laplace_x + laplace_y)
    
    # Enforce boundary conditions
    u_new[0, :] = 1.0   # Bottom boundary
    u_new[-1, :] = 1.0  # Top boundary
    u_new[:, 0] = 1.0   # Left boundary
    u_new[:, -1] = 1.0  # Right boundary
    
    return u_new

# Time marching
for _ in range(nt):
    u = diffusion_2d(u, nu, dx, dy, dt)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Diffusion.npy', u)