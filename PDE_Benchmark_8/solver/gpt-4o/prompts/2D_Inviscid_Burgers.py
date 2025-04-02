import numpy as np

# Parameters
nx, ny = 101, 101  # number of grid points
Lx, Ly = 2.0, 2.0  # domain size
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # grid spacing
dt = 0.001  # time step size
nt = int(0.40 / dt)  # number of time steps

# Initialize the velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Apply initial conditions
u[int(0.5 / dy):int(1.0 / dy + 1), int(0.5 / dx):int(1.0 / dx + 1)] = 2
v[int(0.5 / dy):int(1.0 / dy + 1), int(0.5 / dx):int(1.0 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update u and v using a simple upwind scheme
    u[1:, 1:] = (un[1:, 1:] - 
                 dt / dx * un[1:, 1:] * (un[1:, 1:] - un[1:, :-1]) - 
                 dt / dy * vn[1:, 1:] * (un[1:, 1:] - un[:-1, 1:]))
    
    v[1:, 1:] = (vn[1:, 1:] - 
                 dt / dx * un[1:, 1:] * (vn[1:, 1:] - vn[1:, :-1]) - 
                 dt / dy * vn[1:, 1:] * (vn[1:, 1:] - vn[:-1, 1:]))
    
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
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts/u_2D_Inviscid_Burgers.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts/v_2D_Inviscid_Burgers.npy', v)