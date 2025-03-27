import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 31, 31
nt = 50
nu = 1.0
sigma = 0.25
dx = dy = 2.0 / (nx - 1)
dt = sigma * dx * dy / nu

# Initialize the field
u = np.ones((ny, nx))
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                     nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Save the final solution to a .npy file
np.save('final_solution.npy', u)

# Optional: Visualize the final result
plt.figure(figsize=(8, 6))
plt.contourf(np.linspace(0, 2, nx), np.linspace(0, 2, ny), u, cmap='viridis')
plt.colorbar()
plt.title('2D Diffusion at Final Time Step')
plt.xlabel('x')
plt.ylabel('y')
plt.show()