import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 81, 81
nt = 100
c = 1.0
sigma = 0.2
dx = dy = 2.0 / (nx - 1)
dt = sigma * min(dx, dy) / c

# Initialize the solution array
u = np.ones((ny, nx))

# Initial condition: hat function
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    u[1:, 1:] = (un[1:, 1:] - 
                 c * dt / dx * (un[1:, 1:] - un[1:, :-1]) - 
                 c * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    
    # Enforce boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Save the final solution to a .npy file
np.save('final_solution.npy', u)

# Visualize the final solution
plt.figure(figsize=(8, 6))
plt.contourf(np.linspace(0, 2, nx), np.linspace(0, 2, ny), u, cmap='viridis')
plt.colorbar()
plt.title('Final Solution at t = {:.2f}'.format(nt * dt))
plt.xlabel('x')
plt.ylabel('y')
plt.show()