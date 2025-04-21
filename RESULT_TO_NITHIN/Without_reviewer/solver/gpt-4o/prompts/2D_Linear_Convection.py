import numpy as np

# Parameters
c = 1.0  # Convection speed
nx, ny = 101, 101  # Number of grid points
Lx, Ly = 2.0, 2.0  # Domain size
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # Grid spacing
dt = 0.01  # Time step size
nt = int(0.50 / dt)  # Number of time steps

# Create the grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize the solution array
u = np.ones((ny, nx))

# Apply initial conditions
u[(X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)] = 2.0

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    # Update the solution using upwind scheme
    u[1:, 1:] = (un[1:, 1:] - c * dt / dx * (un[1:, 1:] - un[1:, :-1])
                 - c * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1.0
    u[-1, :] = 1.0
    u[:, 0] = 1.0
    u[:, -1] = 1.0

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_2D_Linear_Convection.npy', u)