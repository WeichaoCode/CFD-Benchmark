import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Domain parameters
Lx, Ly = 2.0, 2.0
nx, ny = 100, 100
nt = 1000
dx, dy = Lx / (nx-1), Ly / (ny-1)
dt = 0.1 / nt

# Physical parameters
rho = 1.0
nu = 0.1
F = 1.0

# Grid generation
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Time-stepping using finite difference method
for n in range(nt):
    # Compute derivatives using central differences
    u_x = np.gradient(u, dx, axis=1)
    u_y = np.gradient(u, dy, axis=0)
    v_x = np.gradient(v, dx, axis=1)
    v_y = np.gradient(v, dy, axis=0)
    
    # Momentum equations
    u_new = u - dt * (u * u_x + v * u_y) + \
            dt * (nu * (np.gradient(u_x, dx, axis=1) + np.gradient(u_y, dy, axis=0)) + F)
    
    v_new = v - dt * (u * v_x + v * v_y) + \
            dt * (nu * (np.gradient(v_x, dx, axis=1) + np.gradient(v_y, dy, axis=0)))
    
    # Pressure Poisson equation
    p_rhs = -rho * (u_x**2 + 2 * u_y * v_x + v_y**2)
    
    # Solve Poisson equation (simplified)
    p_new = np.zeros_like(p)
    for _ in range(50):
        p_new[1:-1, 1:-1] = 0.25 * (p[1:-1, 2:] + p[1:-1, :-2] + 
                                     p[2:, 1:-1] + p[:-2, 1:-1] - 
                                     dx**2 * p_rhs[1:-1, 1:-1])
        
        # Periodic BC in x
        p_new[:, 0] = p_new[:, -2]
        p_new[:, -1] = p_new[:, 1]
        
        # Zero gradient in y
        p_new[0, :] = p_new[1, :]
        p_new[-1, :] = p_new[-2, :]
    
    # Update fields
    u, v, p = u_new, v_new, p_new
    
    # No-slip BC
    u[0, :] = u[-1, :] = 0
    v[0, :] = v[-1, :] = 0

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/u_2D_Navier_Stokes_Channel.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/v_2D_Navier_Stokes_Channel.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/p_2D_Navier_Stokes_Channel.npy', p)