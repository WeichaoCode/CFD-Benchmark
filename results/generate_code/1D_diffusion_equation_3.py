import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
Nx = 101  # Number of spatial points
dx = L/(Nx-1)  # Spatial step size
x = np.linspace(0, L, Nx)  # Spatial grid

T = 2.0  # Final time
nt = 500  # Number of time steps
dt = T/nt  # Time step size
nu = 0.3  # Viscosity

# Initialize solution array
u = np.ones(Nx)
u[int(0.5/dx):int(1.0/dx)+1] = 2.0

# Stability check
r = nu*dt/dx**2
if r > 0.5:
    print("Warning: Solution may be unstable")

# Time stepping
for n in range(nt):
    un = u.copy()
    
    # Interior points
    for i in range(1, Nx-1):
        u[i] = un[i] + nu*dt/dx**2 * (un[i+1] - 2*un[i] + un[i-1])
    
    # Boundary conditions (Neumann)
    u[0] = u[1]
    u[-1] = u[-2]

# Plot results
plt.figure(figsize=(10,6))
plt.plot(x, u, 'b-', label='t = T')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()