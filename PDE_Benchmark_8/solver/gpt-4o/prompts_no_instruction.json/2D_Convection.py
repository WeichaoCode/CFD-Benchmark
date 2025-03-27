import numpy as np

# Parameters
nx = 101
ny = 101
nt = 80
sigma = 0.2
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = sigma * dx

# Initialize the velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Apply initial conditions
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update the velocity fields using finite difference scheme
    u[1:, 1:] = (un[1:, 1:] - 
                 un[1:, 1:] * dt / dx * (un[1:, 1:] - un[1:, :-1]) - 
                 vn[1:, 1:] * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    
    v[1:, 1:] = (vn[1:, 1:] - 
                 un[1:, 1:] * dt / dx * (vn[1:, 1:] - vn[1:, :-1]) - 
                 vn[1:, 1:] * dt / dy * (vn[1:, 1:] - vn[:-1, 1:]))
    
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
np.save('final_solution_u.npy', u)
np.save('final_solution_v.npy', v)