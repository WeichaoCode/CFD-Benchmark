import numpy as np
import matplotlib.pyplot as plt

# Problem Parameters
Lx, Ly = 2.0, 2.0  # Domain size
nx, ny = 100, 100  # Grid points
nt = 200  # Time steps
dt = 0.4 / nt  # Time step size
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Initialize grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize solution arrays
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial Condition
mask = ((X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0))
u[mask] = 2
v[mask] = 2

# Finite Volume Method with Lax-Wendroff scheme
def lax_wendroff_2d(u, v, dx, dy, dt):
    # Compute fluxes
    u_new = np.copy(u)
    v_new = np.copy(v)
    
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # u-momentum
            u_flux_x_plus = 0.5 * (u[i,j+1] + u[i,j]) - 0.5 * dt/dx * (u[i,j+1]**2 - u[i,j]**2)
            u_flux_x_minus = 0.5 * (u[i,j] + u[i,j-1]) - 0.5 * dt/dx * (u[i,j]**2 - u[i,j-1]**2)
            
            u_flux_y_plus = 0.5 * (v[i,j] + v[i+1,j]) * u[i,j] - 0.5 * dt/dy * (v[i+1,j] * u[i+1,j] - v[i,j] * u[i,j])
            u_flux_y_minus = 0.5 * (v[i,j-1] + v[i+1,j-1]) * u[i,j-1] - 0.5 * dt/dy * (v[i+1,j-1] * u[i+1,j-1] - v[i,j-1] * u[i,j-1])
            
            u_new[i,j] = u[i,j] - dt/dx * (u_flux_x_plus - u_flux_x_minus) - dt/dy * (u_flux_y_plus - u_flux_y_minus)
            
            # v-momentum
            v_flux_x_plus = 0.5 * (u[i,j] + u[i,j+1]) * v[i,j] - 0.5 * dt/dx * (u[i,j+1] * v[i,j+1] - u[i,j] * v[i,j])
            v_flux_x_minus = 0.5 * (u[i,j-1] + u[i,j]) * v[i,j-1] - 0.5 * dt/dx * (u[i,j] * v[i,j] - u[i,j-1] * v[i,j-1])
            
            v_flux_y_plus = 0.5 * (v[i+1,j] + v[i,j]) - 0.5 * dt/dy * (v[i+1,j]**2 - v[i,j]**2)
            v_flux_y_minus = 0.5 * (v[i,j] + v[i-1,j]) - 0.5 * dt/dy * (v[i,j]**2 - v[i-1,j]**2)
            
            v_new[i,j] = v[i,j] - dt/dx * (v_flux_x_plus - v_flux_x_minus) - dt/dy * (v_flux_y_plus - v_flux_y_minus)
    
    # Boundary conditions
    u_new[0,:] = 1
    u_new[-1,:] = 1
    u_new[:,0] = 1
    u_new[:,-1] = 1
    
    v_new[0,:] = 1
    v_new[-1,:] = 1
    v_new[:,0] = 1
    v_new[:,-1] = 1
    
    return u_new, v_new

# Time integration
for _ in range(nt):
    u, v = lax_wendroff_2d(u, v, dx, dy, dt)

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_2D_Inviscid_Burgers.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/v_2D_Inviscid_Burgers.npy', v)