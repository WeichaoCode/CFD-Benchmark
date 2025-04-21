import numpy as np

# Grid parameters
nx = 100  # Number of points in x
ny = 100  # Number of points in y
nt = 1000  # Number of timesteps
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
dt = 0.0003

# Initialize arrays
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set initial conditions
u[(Y >= 0.5) & (Y <= 1) & (X >= 0.5) & (X <= 1)] = 2
v[(Y >= 0.5) & (Y <= 1) & (X >= 0.5) & (X <= 1)] = 2

# Time stepping
for n in range(nt):
    # Store previous values
    un = u.copy()
    vn = v.copy()
    
    # Update interior points
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                     dt * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2])/dx -
                     dt * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1])/dy)
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     dt * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])/dx -
                     dt * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])/dy)
    
    # Apply boundary conditions
    u[0, :] = 1  # Bottom
    u[-1, :] = 1 # Top
    u[:, 0] = 1  # Left
    u[:, -1] = 1 # Right
    
    v[0, :] = 1  # Bottom
    v[-1, :] = 1 # Top
    v[:, 0] = 1  # Left
    v[:, -1] = 1 # Right

# Save final solutions
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Convection.npy', u)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/v_2D_Convection.npy', v)