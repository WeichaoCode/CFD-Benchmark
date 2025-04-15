import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
L = 2.0       # Length of the domain
T = 1.0       # Time of simulation
nx = 80       # Number of spatial points in the domain
nt = 100      # Number of time steps
c = 1.0       # Wave speed
dx = L/nx     # Spatial discretization size
dt = T/nt     # Time discretization size
CFL = c*dt/dx # CFL number

# Step 2: Discretize space and time
x = np.linspace(0, L, nx)  # Define spatial grid
t = np.linspace(0, T, nt)  # Define time grid

# Step 3: Set initial wave profile
u_storage = np.zeros((nt,nx)) 
u_storage[0,:] = np.piecewise(x, [x < 0.5, x >= 0.5], [1.0, 0.0])

# Step 4: Iterate using finite difference scheme
for t in range(1, nt):
    u_storage[t, 1:-1] = u_storage[t-1, 1:-1] - CFL * 0.5 * (u_storage[t-1, 2:] - u_storage[t-1, :-2])

# Step 5: Plot the wave evolution
for i in range(nt):
    if i % 20 == 0: # Only plot every 20th time step for clarity
        plt.plot(x, u_storage[i,:])
plt.show()