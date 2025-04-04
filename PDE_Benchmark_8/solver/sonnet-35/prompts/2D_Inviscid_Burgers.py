import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
Lx, Ly = 2.0, 2.0  # Domain dimensions
nx, ny = 100, 100  # Number of grid points
nt = 200  # Number of time steps
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = 0.002  # Time step size

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize solution arrays
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial condition
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2
v[mask] = 2

# Finite difference method (simplified explicit scheme)
def burgers_2d_step(u, v):
    un = u.copy()
    vn = v.copy()
    
    # Interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # u-momentum equation
            u[i,j] = un[i,j] - un[i,j] * dt/dx * (un[i,j] - un[i,j-1]) \
                              - vn[i,j] * dt/dy * (un[i,j] - un[i-1,j])
            
            # v-momentum equation  
            v[i,j] = vn[i,j] - un[i,j] * dt/dx * (vn[i,j] - vn[i,j-1]) \
                              - vn[i,j] * dt/dy * (vn[i,j] - vn[i-1,j])
    
    # Boundary conditions
    u[0,:] = 1
    u[-1,:] = 1
    u[:,0] = 1
    u[:,-1] = 1
    
    v[0,:] = 1
    v[-1,:] = 1
    v[:,0] = 1
    v[:,-1] = 1
    
    return u, v

# Time marching
for n in range(nt):
    u, v = burgers_2d_step(u, v)

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/u_2D_Inviscid_Burgers.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/v_2D_Inviscid_Burgers.npy', v)