import numpy as np

# Parameters
nx, ny = 41, 41
nt = 120
sigma = 0.0009
nu = 0.01
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = sigma * dx * dy / nu

# Initialize the grid
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Initialize velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Apply initial conditions: u = v = 2 for 0.5 <= x, y <= 1
condition = np.logical_and.reduce((X >= 0.5, X <= 1.0, Y >= 0.5, Y <= 1.0))
u[condition] = 2
v[condition] = 2

# Time-stepping loop
for _ in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update u
    u[1:-1,1:-1] = (
        un[1:-1,1:-1]
        - un[1:-1,1:-1] * (dt/dx) * (un[1:-1,1:-1] - un[0:-2,1:-1])
        - vn[1:-1,1:-1] * (dt/dy) * (un[1:-1,1:-1] - un[1:-1,0:-2])
        + nu * (
            (dt/dx**2) * (un[2:,1:-1] - 2 * un[1:-1,1:-1] + un[0:-2,1:-1]) +
            (dt/dy**2) * (un[1:-1,2:] - 2 * un[1:-1,1:-1] + un[1:-1,0:-2])
        )
    )
    
    # Update v
    v[1:-1,1:-1] = (
        vn[1:-1,1:-1]
        - un[1:-1,1:-1] * (dt/dx) * (vn[1:-1,1:-1] - vn[0:-2,1:-1])
        - vn[1:-1,1:-1] * (dt/dy) * (vn[1:-1,1:-1] - vn[1:-1,0:-2])
        + nu * (
            (dt/dx**2) * (vn[2:,1:-1] - 2 * vn[1:-1,1:-1] + vn[0:-2,1:-1]) +
            (dt/dy**2) * (vn[1:-1,2:] - 2 * vn[1:-1,1:-1] + vn[1:-1,0:-2])
        )
    )
    
    # Apply Dirichlet boundary conditions: u = v = 1 on all boundaries
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Save the final velocity fields
np.save('u.npy', u)
np.save('v.npy', v)