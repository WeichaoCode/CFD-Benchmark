import numpy as np

# Parameters
nx, ny = 101, 101
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
dx = 2.0 / (nx - 1)
dy = dx
sigma = 0.2
dt = sigma * dx
nt = 80

# Initialize u and v
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Apply initial conditions
X, Y = np.meshgrid(x, y)
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2.0
v[mask] = 2.0

# Time-stepping loop
for _ in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # First-order upwind scheme for u
    u[1:-1,1:-1] = un[1:-1,1:-1] - dt * (
        un[1:-1,1:-1] * (un[1:-1,1:-1] - un[1:-1,0:-2]) / dx +
        vn[1:-1,1:-1] * (un[1:-1,1:-1] - un[0:-2,1:-1]) / dy
    )
    
    # First-order upwind scheme for v
    v[1:-1,1:-1] = vn[1:-1,1:-1] - dt * (
        un[1:-1,1:-1] * (vn[1:-1,1:-1] - vn[1:-1,0:-2]) / dx +
        vn[1:-1,1:-1] * (vn[1:-1,1:-1] - vn[0:-2,1:-1]) / dy
    )
    
    # Apply boundary conditions
    u[0, :] = 1.0
    u[-1, :] = 1.0
    u[:, 0] = 1.0
    u[:, -1] = 1.0
    v[0, :] = 1.0
    v[-1, :] = 1.0
    v[:, 0] = 1.0
    v[:, -1] = 1.0

# Save the final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/u_2D_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/v_2D_Convection.npy', v)