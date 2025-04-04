import numpy as np

# Domain parameters
Lx, Ly = 2.0, 2.0
nx, ny = 100, 100
dx, dy = Lx / (nx-1), Ly / (ny-1)
nt = 100
CFL = 0.5  # Courant-Friedrichs-Lewy condition

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initial conditions
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set initial condition in specified region
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2.0
v[mask] = 2.0

# Boundary conditions 
def apply_boundary_conditions(u, v):
    u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = 1.0
    v[0,:] = v[-1,:] = v[:,0] = v[:,-1] = 1.0
    return u, v

# Finite difference method with flux limiter (Superbee limiter)
def minmod(a, b):
    return np.sign(a) * np.maximum(0, np.minimum(np.abs(a), np.sign(a) * b))

def solve_advection(u, v):
    u_new = u.copy()
    v_new = v.copy()
    
    # Compute time step using CFL condition
    u_max = np.max(np.abs(u))
    v_max = np.max(np.abs(v))
    dt = CFL * min(dx/u_max, dy/v_max)
    
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # Compute flux limiters for u
            r_u_x = ((u[i,j] - u[i,j-1]) / (u[i,j+1] - u[i,j] + 1e-10))
            r_u_y = ((u[i,j] - u[i-1,j]) / (u[i+1,j] - u[i,j] + 1e-10))
            
            phi_u_x = minmod(1, r_u_x)
            phi_u_y = minmod(1, r_u_y)
            
            # Compute flux limiters for v
            r_v_x = ((v[i,j] - v[i,j-1]) / (v[i,j+1] - v[i,j] + 1e-10))
            r_v_y = ((v[i,j] - v[i-1,j]) / (v[i+1,j] - v[i,j] + 1e-10))
            
            phi_v_x = minmod(1, r_v_x)
            phi_v_y = minmod(1, r_v_y)
            
            # Compute antidiffusive fluxes
            u_flux_x = 0.5 * dt/dx * u[i,j] * (u[i,j+1] - u[i,j-1]) * phi_u_x
            u_flux_y = 0.5 * dt/dy * v[i,j] * (u[i+1,j] - u[i-1,j]) * phi_u_y
            
            v_flux_x = 0.5 * dt/dx * u[i,j] * (v[i,j+1] - v[i,j-1]) * phi_v_x
            v_flux_y = 0.5 * dt/dy * v[i,j] * (v[i+1,j] - v[i-1,j]) * phi_v_y
            
            # Update solution
            u_new[i,j] = u[i,j] - (u_flux_x + u_flux_y)
            v_new[i,j] = v[i,j] - (v_flux_x + v_flux_y)
    
    return u_new, v_new

# Time integration
for _ in range(nt):
    u, v = solve_advection(u, v)
    u, v = apply_boundary_conditions(u, v)

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/u_2D_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/v_2D_Convection.npy', v)