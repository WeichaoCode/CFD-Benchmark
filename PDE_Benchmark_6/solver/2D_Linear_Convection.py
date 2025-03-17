import numpy as np
import matplotlib.pyplot as plt

# Define the grid parameters
nx, ny = 81, 81  # Grid resolution
lx, ly = 2.0, 2.0  # Domain size
dx, dy = lx / (nx - 1), ly / (ny - 1)  # Grid spacing
c = 1.0  # Convection speed
sigma = 0.2  # Stability parameter
dt = sigma * min(dx, dy) / c  # Time step
nt = 100  # Number of time steps

# Initialize solution matrix
u = np.ones([ny, nx])  # Initialize u with ones
# Set initial condition
u[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2

# Apply numerical method
for n in range(nt):
    un = u.copy()  
    # NB: boundary conditions are automatically handled by providing a slice of u where the respective boundary values are excluded
    u[1:, 1:] = (un[1:, 1:] 
                 - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) 
                 - (c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))

# Save the final result into .npy file
np.save("solution.npy", u)

# Plot the result
X, Y = np.meshgrid(np.linspace(0, lx, nx), np.linspace(0, ly, ny))
plt.contourf(X, Y, u)
plt.colorbar()
plt.title('2D Linear Convection')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()