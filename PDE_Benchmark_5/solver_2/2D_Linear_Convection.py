import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Lx = Ly = 2    # domain size
nx = ny = 101  # number of points in each direction
nt = 200       # number of time steps
c = 1.0        # wave speed

# Discretize space and time
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = 0.02      
# Ensure dt is small enough to satisfy the stability criterion (CFL condition)

# Initialize field
u = np.ones((ny, nx))  # 2D array for u
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2  # initial condition

# Initialize temporary array
un = np.ones((ny, nx))

# Evolve field in time
for n in range(nt):
    un = u.copy()
    u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1]))
                 - (c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))

    # Apply boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
  
# Plot result
plt.imshow(u)
plt.colorbar()
plt.title("2D Linear Convection")
plt.show()