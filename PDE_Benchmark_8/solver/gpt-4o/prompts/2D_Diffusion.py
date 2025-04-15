import numpy as np

# Parameters
nu = 1.0
Lx, Ly = 2.0, 2.0
Nx, Ny = 101, 101
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
dt = 0.0001  # Reduced time step for stability
T_final = 0.3777

# Stability condition for explicit scheme
# dt <= 0.5 * min(dx^2, dy^2) / nu
dt = min(dt, 0.5 * min(dx**2, dy**2) / nu)

# Create the grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
u = np.ones((Nx, Ny))

# Initial condition
u[int(0.5/dx):int(1.0/dx)+1, int(0.5/dy):int(1.0/dy)+1] = 2

# Time-stepping loop
t = 0.0
while t < T_final:
    un = u.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                     nu * dt / dx**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) +
                     nu * dt / dy**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    t += dt

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts/u_2D_Diffusion.npy', u)