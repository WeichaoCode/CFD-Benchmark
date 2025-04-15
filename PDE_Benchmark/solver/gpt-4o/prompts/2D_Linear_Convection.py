import numpy as np

# Parameters
c = 1.0  # Convection speed
Lx, Ly = 2.0, 2.0  # Domain size
Nx, Ny = 101, 101  # Number of grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
dt = 0.005  # Time step size
T = 0.50  # Final time
nt = int(T / dt)  # Number of time steps

# Create grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition
u = np.ones((Ny, Nx))
u[(X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)] = 2.0

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    # Update u using upwind scheme
    u[1:, 1:] = (un[1:, 1:] - c * dt / dx * (un[1:, 1:] - un[1:, :-1])
                 - c * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    
    # Apply boundary conditions
    u[0, :] = 1.0  # y = 0
    u[-1, :] = 1.0  # y = 2
    u[:, 0] = 1.0  # x = 0
    u[:, -1] = 1.0  # x = 2

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_2D_Linear_Convection.npy', u)