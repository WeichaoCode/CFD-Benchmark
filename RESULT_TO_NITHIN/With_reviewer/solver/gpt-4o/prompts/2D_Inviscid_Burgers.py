import numpy as np

# Parameters
nx, ny = 101, 101  # number of grid points
nt = 100  # number of time steps
dx = 2.0 / (nx - 1)
dy = 2.0 / (ny - 1)
dt = 0.004  # time step size
c = 1  # wave speed

# Initialize the velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial conditions
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update the velocity fields using finite difference method
    u[1:, 1:] = (un[1:, 1:] - 
                 dt / dx * un[1:, 1:] * (un[1:, 1:] - un[1:, :-1]) - 
                 dt / dy * vn[1:, 1:] * (un[1:, 1:] - un[:-1, 1:]))
    
    v[1:, 1:] = (vn[1:, 1:] - 
                 dt / dx * un[1:, 1:] * (vn[1:, 1:] - vn[1:, :-1]) - 
                 dt / dy * vn[1:, 1:] * (vn[1:, 1:] - vn[:-1, 1:]))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Save the final solution
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_2D_Inviscid_Burgers.npy', u)
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/v_2D_Inviscid_Burgers.npy', v)