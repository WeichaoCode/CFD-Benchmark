import numpy as np

# Grid parameters
nx = ny = 101
dx = dy = 2.0/(nx-1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Time parameters
dt = 0.0001
t_final = 0.3777
nt = int(t_final/dt)

# Physical parameters
nu = 0.05

# Initialize solution array
u = np.ones((ny, nx))

# Set initial condition
u[(X >= 0.5) & (X <= 1) & (Y >= 0.5) & (Y <= 1)] = 2

# Time stepping
for n in range(nt):
    un = u.copy()
    
    # Interior points
    u[1:-1,1:-1] = un[1:-1,1:-1] + nu*dt*(
        (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])/dx**2 +
        (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1])/dy**2
    )
    
    # Boundary conditions
    u[0,:] = 1  # Bottom
    u[-1,:] = 1 # Top
    u[:,0] = 1  # Left
    u[:,-1] = 1 # Right

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Diffusion.npy', u)