import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
Lx, Ly = 2.0, 2.0
nx, ny = 100, 100
dx, dy = Lx / (nx-1), Ly / (ny-1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Time parameters
T = 0.40
nt = 200
dt = T / nt

# Initial conditions
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set initial condition for central region
u[(y >= 0.5) & (y <= 1), (x >= 0.5) & (x <= 1)] = 2
v[(y >= 0.5) & (y <= 1), (x >= 0.5) & (x <= 1)] = 2

# Boundary conditions
u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = 1
v[0,:] = v[-1,:] = v[:,0] = v[:,-1] = 1

# Central difference scheme with forward Euler time stepping
def burgers_2d(u, v):
    un = u.copy()
    vn = v.copy()
    
    # Interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # u-momentum
            u[i,j] = un[i,j] - un[i,j] * dt/(2*dx) * (un[i,j+1] - un[i,j-1]) \
                              - vn[i,j] * dt/(2*dy) * (un[i+1,j] - un[i-1,j])
            
            # v-momentum  
            v[i,j] = vn[i,j] - un[i,j] * dt/(2*dx) * (vn[i,j+1] - vn[i,j-1]) \
                              - vn[i,j] * dt/(2*dy) * (vn[i+1,j] - vn[i-1,j])
    
    return u, v

# Time integration
for _ in range(nt):
    u, v = burgers_2d(u, v)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/u_2D_Inviscid_Burgers.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/v_2D_Inviscid_Burgers.npy', v)