import numpy as np
import matplotlib.pyplot as plt

# Parameters
Nx = 101  # Number of grid points
L = 2.0   # Length of domain
dx = L/(Nx-1)  # Grid spacing
c = 1.0   # Wave speed
T = 2.0   # Final time
nt = 500  # Number of time steps
dt = T/nt # Time step size
CFL = c*dt/dx  # CFL number

# Grid points
x = np.linspace(0, L, Nx)

# Initial condition
u = np.ones(Nx)
u[(x>=0.5) & (x<=1.0)] = 2.0
u_init = u.copy()

# Time stepping
for n in range(nt):
    un = u.copy()
    # First-order upwind scheme
    u[1:] = un[1:] - CFL*(un[1:] - un[:-1])
    # Periodic BC
    u[0] = u[-1]

# Plot results
plt.figure(figsize=(10,6))
plt.plot(x, u_init, 'b-', label='Initial')
plt.plot(x, u, 'r--', label='Final')
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Linear Convection')
plt.legend()
plt.grid(True)
plt.show()