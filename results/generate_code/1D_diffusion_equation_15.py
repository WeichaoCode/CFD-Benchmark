import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0
Nx = 101
dx = L/(Nx-1)
x = np.linspace(0, L, Nx)
nu = 0.3

T = 2.0
nt = 500
dt = T/nt

# Initialize solution array
u = np.ones(Nx)
u[(x>=0.5) & (x<=1.0)] = 2.0
u_new = np.zeros(Nx)

# Time stepping
for n in range(nt):
    # Interior points
    for i in range(1, Nx-1):
        u_new[i] = u[i] + nu*dt/(dx*dx)*(u[i+1] - 2*u[i] + u[i-1])
    
    # Boundary conditions (Neumann)
    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]
    
    # Update solution
    u = u_new.copy()

# Plot results
plt.figure(figsize=(10,6))
plt.plot(x, u, 'b-', label='Final solution')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('u')
plt.title(f'1D Diffusion at t = {T}')
plt.legend()
plt.show()