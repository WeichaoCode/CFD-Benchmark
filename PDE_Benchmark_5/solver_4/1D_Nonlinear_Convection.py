import numpy as np
import matplotlib.pyplot as plt

# Step 1: Defining parameters

L = 2.0   # length of the domain
T = 1.0   # final time
nx = 101  # number of spatial points in the domain
nt = 100  # number of time steps
dx = L / (nx - 1)
dt = T / nt
cfl = dt / dx # Courant number

# Step 2: Spatial discretization
x = np.linspace(0, L, nx)

# Step 3: Defining initial wave profile 
u = np.where((0.5 <= x) & (x <= 1), 2, 1)

# Step 4:  Time stepping 
for n in range(nt):
    un = u.copy()
    for i in range(1, nx):
        # Apply upwind scheme
        u[i] = un[i] - cfl * un[i] * (un[i] - un[i - 1])

# Step 5:  Plotting the wave evolution
plt.figure(figsize=(7, 5))
plt.plot(x, u, 'b-', linewidth=2)
plt.title('1D Nonlinear Convection')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.show()