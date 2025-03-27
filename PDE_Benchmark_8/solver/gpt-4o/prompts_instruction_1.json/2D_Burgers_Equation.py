import numpy as np

# Parameters
nx, ny = 41, 41
nt = 120
nu = 0.01
sigma = 0.0009
dx = dy = 2 / (nx - 1)
dt = sigma * dx * dy / nu

# Initialize the velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial condition: u = v = 2 for 0.5 <= x, y <= 1
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update u and v using finite difference method
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1]) +
                     nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                     nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) +
                     nu * dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                     nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]))

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