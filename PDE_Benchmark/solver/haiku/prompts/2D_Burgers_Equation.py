import numpy as np

# Parameters
nx = 50  # Number of points in x
ny = 50  # Number of points in y
nt = 100  # Number of timesteps
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
dt = 0.027/nt
nu = 0.01

# Initialize arrays
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set initial conditions
u[(Y >= 0.5) & (Y <= 1) & (X >= 0.5) & (X <= 1)] = 2
v[(Y >= 0.5) & (Y <= 1) & (X >= 0.5) & (X <= 1)] = 2

un = np.zeros((ny, nx))
vn = np.zeros((ny, nx))

# Time stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Interior points
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                     dt * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2])/dx -
                     dt * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1])/dy +
                     nu * dt * ((un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, 0:-2])/dx**2 +
                               (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[0:-2, 1:-1])/dy**2))
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     dt * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])/dx -
                     dt * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])/dy +
                     nu * dt * ((vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, 0:-2])/dx**2 +
                               (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[0:-2, 1:-1])/dy**2))
    
    # Boundary conditions
    u[0, :] = 1  # Bottom
    u[-1, :] = 1 # Top
    u[:, 0] = 1  # Left
    u[:, -1] = 1 # Right
    
    v[0, :] = 1  # Bottom
    v[-1, :] = 1 # Top
    v[:, 0] = 1  # Left
    v[:, -1] = 1 # Right

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Burgers_Equation.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/v_2D_Burgers_Equation.npy', v)