import numpy as np

# Parameters
Lx = Ly = 2.0  # Domain size
Nx = Ny = 100  # Number of grid points
dx = Lx/(Nx-1)  # Grid spacing
dy = Ly/(Ny-1)
c = 1.0  # Convection speed
T = 0.50  # Final time
dt = 0.001  # Time step
Nt = int(T/dt)  # Number of time steps

# Initialize grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize solution array
u = np.ones((Ny, Nx))

# Set initial conditions
u[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2.0

# Time stepping
for n in range(Nt):
    # Create copy of solution for previous time step
    u_old = u.copy()
    
    # Update interior points using upwind scheme
    u[1:-1, 1:-1] = u_old[1:-1, 1:-1] - dt * c * (
        (u_old[1:-1, 1:-1] - u_old[1:-1, :-2])/dx +  # x-derivative
        (u_old[1:-1, 1:-1] - u_old[:-2, 1:-1])/dy    # y-derivative
    )
    
    # Apply boundary conditions
    u[0, :] = 1.0  # Bottom
    u[-1, :] = 1.0  # Top
    u[:, 0] = 1.0  # Left
    u[:, -1] = 1.0  # Right

# Save final solution
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Linear_Convection.npy', u)