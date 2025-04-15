import numpy as np

# Grid parameters
nx = ny = 50
dx = dy = 2.0/(nx-1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Time parameters
dt = 0.5 * min(dx,dy)**2  # For stability
t_final = 0.3777
nt = int(t_final/dt)

# Initialize solution
u = np.ones((ny, nx))
u[(Y>=0.5) & (Y<=1.0) & (X>=0.5) & (X<=1.0)] = 2.0

# Diffusion coefficient
nu = 1.0

# Time stepping
for n in range(nt):
    un = u.copy()
    
    # Interior points
    u[1:-1,1:-1] = un[1:-1,1:-1] + nu*dt*(
        (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2])/dx**2 +
        (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1])/dy**2
    )
    
    # Boundary conditions
    u[0,:] = 1  # Bottom
    u[-1,:] = 1 # Top
    u[:,0] = 1  # Left
    u[:,-1] = 1 # Right

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Diffusion.npy', u)