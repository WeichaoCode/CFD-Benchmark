import numpy as np

# Parameters
nx, ny = 101, 101
nt = 80
sigma = 0.2
dx = dy = 2 / (nx - 1)
dt = sigma * dx

# Initialize the velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial condition: u = v = 2 for 0.5 <= x, y <= 1
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

u[(X >= 0.5) & (X <= 1) & (Y >= 0.5) & (Y <= 1)] = 2
v[(X >= 0.5) & (X <= 1) & (Y >= 0.5) & (Y <= 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update u using 1st-order upwind scheme
    u[1:, 1:] = (un[1:, 1:] - 
                 dt / dx * un[1:, 1:] * (un[1:, 1:] - un[1:, :-1]) - 
                 dt / dy * vn[1:, 1:] * (un[1:, 1:] - un[:-1, 1:]))
    
    # Update v using 1st-order upwind scheme
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
np.save('final_solution_u.npy', u)
np.save('final_solution_v.npy', v)