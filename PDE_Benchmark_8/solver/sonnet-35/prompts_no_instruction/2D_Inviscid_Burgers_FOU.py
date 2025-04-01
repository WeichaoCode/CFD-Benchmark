import numpy as np

# Grid parameters
nx, ny = 151, 151
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

# Time parameters
nt = 300
sigma = 0.2
dt = sigma * min(dx, dy) / 2

# Initialize solution arrays
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial conditions
u[(0.5 <= x) & (x <= 1), (0.5 <= y) & (y <= 1)] = 2
v[(0.5 <= x) & (x <= 1), (0.5 <= y) & (y <= 1)] = 2

# Boundary conditions
u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 1
v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 1

# Solution loop
for _ in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # First-Order Upwind scheme
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            u[i, j] = un[i, j] - un[i, j] * dt/dx * (un[i, j] - un[i, j-1]) \
                               - vn[i, j] * dt/dy * (un[i, j] - un[i-1, j])
            
            v[i, j] = vn[i, j] - un[i, j] * dt/dx * (vn[i, j] - vn[i, j-1]) \
                               - vn[i, j] * dt/dy * (vn[i, j] - vn[i-1, j])

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_2D_Inviscid_Burgers_FOU.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/v_2D_Inviscid_Burgers_FOU.npy', v)