import numpy as np

# Grid parameters
nx = 100  # Number of points in x
ny = 100  # Number of points in y
nt = 1000  # Number of timesteps
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
dt = 0.0005
c = 1.0  # Convection speed

# Initialize arrays
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)
u = np.ones((ny, nx))

# Set initial conditions
u[(Y>=0.5) & (Y<=1.0) & (X>=0.5) & (X<=1.0)] = 2.0

# Time stepping
for n in range(nt):
    un = u.copy()
    
    # Interior points
    u[1:-1,1:-1] = un[1:-1,1:-1] - c*dt/dx*(un[1:-1,1:-1]-un[1:-1,:-2]) \
                                 - c*dt/dy*(un[1:-1,1:-1]-un[:-2,1:-1])
    
    # Boundary conditions
    u[0,:] = 1  # Bottom
    u[-1,:] = 1 # Top
    u[:,0] = 1  # Left
    u[:,-1] = 1 # Right

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Linear_Convection.npy', u)