import numpy as np

# Parameters
nx, ny = 101, 101
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
CFL = 0.2
u_max = 2
v_max = 2
dt = CFL * min(dx, dy) / max(u_max, v_max)
nt = int(0.32 / dt) + 1
dt = 0.32 / nt

# Initialize fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Create meshgrid for initial condition
X, Y = np.meshgrid(x, y)
mask = (X >= 0.5) & (X <=1) & (Y >=0.5) & (Y <=1)
u[mask] = 2
v[mask] = 2

# Time-stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Compute derivatives using backward differences (assuming u, v > 0)
    u_x = (un[1:-1,1:-1] - un[1:-1, :-2]) / dx
    u_y = (un[1:-1,1:-1] - un[:-2,1:-1]) / dy
    v_x = (vn[1:-1,1:-1] - vn[1:-1, :-2]) / dx
    v_y = (vn[1:-1,1:-1] - vn[:-2,1:-1]) / dy
    
    # Update interior points
    u[1:-1,1:-1] = un[1:-1,1:-1] - dt * (un[1:-1,1:-1] * u_x + vn[1:-1,1:-1] * u_y)
    v[1:-1,1:-1] = vn[1:-1,1:-1] - dt * (un[1:-1,1:-1] * v_x + vn[1:-1,1:-1] * v_y)
    
    # Apply boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

    # Prevent numerical overflow
    u = np.clip(u, -1e10, 1e10)
    v = np.clip(v, -1e10, 1e10)

# Save final solution
np.save('u.npy', u)
np.save('v.npy', v)