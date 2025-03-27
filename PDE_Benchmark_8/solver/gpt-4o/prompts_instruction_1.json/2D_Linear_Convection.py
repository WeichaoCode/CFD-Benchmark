import numpy as np

# Parameters
nx, ny = 81, 81
nt = 100
c = 1.0
sigma = 0.2

# Domain
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = sigma * min(dx, dy) / c

# Initial condition
u = np.ones((ny, nx))
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    for j in range(1, ny):
        for i in range(1, nx):
            u[j, i] = (un[j, i] 
                       - c * dt / dx * (un[j, i] - un[j, i-1]) 
                       - c * dt / dy * (un[j, i] - un[j-1, i]))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Save the final solution
np.save('final_solution.npy', u)