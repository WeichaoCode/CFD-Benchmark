import numpy as np

# Parameters
nx, ny = 101, 101
nt = 80
Lx, Ly = 2.0, 2.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
sigma = 0.2
dt = sigma * dx

# Initialize the velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial condition: u = v = 2 for 0.5 <= x, y <= 1
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

u[(X >= 0.5) & (X <= 1) & (Y >= 0.5) & (Y <= 1)] = 2
v[(X >= 0.5) & (X <= 1) & (Y >= 0.5) & (Y <= 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update u and v using finite difference scheme
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1]))
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]))
    
    # Apply Dirichlet boundary conditions
    u[:, 0] = 1
    u[:, -1] = 1
    u[0, :] = 1
    u[-1, :] = 1
    
    v[:, 0] = 1
    v[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_2D_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/v_2D_Convection.npy', v)