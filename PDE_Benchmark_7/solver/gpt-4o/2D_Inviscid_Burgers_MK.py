import numpy as np
import matplotlib.pyplot as plt

# Define computational parameters
Lx, Ly = 2.0, 2.0
nx, ny = 151, 151
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
sigma = 0.2
dt = sigma * min(dx, dy)
nt = 300

# Initialize velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Apply initial condition: hat function
u[int(0.5/dy):int(1.0/dy + 1), int(0.5/dx):int(1.0/dx + 1)] = 2
v[int(0.5/dy):int(1.0/dy + 1), int(0.5/dx):int(1.0/dx + 1)] = 2

# Solver using First-Order Upwind Method
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update u and v using the upwind scheme
    u[1:, 1:] = (un[1:, 1:] - 
                 dt * un[1:, 1:] * (un[1:, 1:] - un[:-1, 1:]) / dx - 
                 dt * vn[1:, 1:] * (un[1:, 1:] - un[1:, :-1]) / dy)
                 
    v[1:, 1:] = (vn[1:, 1:] - 
                 dt * un[1:, 1:] * (vn[1:, 1:] - vn[:-1, 1:]) / dx - 
                 dt * vn[1:, 1:] * (vn[1:, 1:] - vn[1:, :-1]) / dy)

    # Apply boundary conditions
    u[0, :], u[:, 0] = 1, 1
    u[-1, :], u[:, -1] = 1, 1
    v[0, :], v[:, 0] = 1, 1
    v[-1, :], v[:, -1] = 1, 1

# Save the final velocity fields
np.save('u_velocity.npy', u)
np.save('v_velocity.npy', v)

# Visualization using quiver plot
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 8))
plt.quiver(X, Y, u, v, scale=5)
plt.title('Velocity Field')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.show()