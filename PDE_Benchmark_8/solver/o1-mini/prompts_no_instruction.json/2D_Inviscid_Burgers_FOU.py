import numpy as np

# Parameters
nx, ny = 151, 151
x_start, x_end = 0, 2
y_start, y_end = 0, 2
dx = (x_end - x_start) / (nx - 1)
dy = (y_end - y_start) / (ny - 1)
sigma = 0.2
dt = sigma * min(dx, dy) / 2
nt = 300

# Grid
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
X, Y = np.meshgrid(x, y)

# Initial Conditions
u = np.ones((ny, nx))
v = np.ones((ny, nx))
condition = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[condition] = 2.0
v[condition] = 2.0

# Time-stepping
for _ in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # First-order Upwind scheme for u
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                     dt * (un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) / dx +
                           vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) / dy))
    
    # First-order Upwind scheme for v
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - 
                     dt * (un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) / dx +
                           vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) / dy))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = u[-1, :] = 1
    u[:, 0] = u[:, -1] = 1
    v[0, :] = v[-1, :] = 1
    v[:, 0] = v[:, -1] = 1

# Save the final solution
np.save('u.npy', u)
np.save('v.npy', v)