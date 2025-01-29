import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx = 101
x = np.linspace(0, 2, nx)
dx = x[1] - x[0]

# Time parameters
nt = 500
T = 2
dt = T/nt

# Initial condition
u = np.ones(nx)
u[(x >= 0.5) & (x <= 1)] = 2

# Storage for solution
u_sol = np.zeros((nt+1, nx))
u_sol[0] = u.copy()

# Time stepping
for n in range(nt):
    un = u.copy()
    
    # Spatial derivatives using central difference
    dudx = np.zeros_like(u)
    dudx[1:-1] = (u[2:] - u[:-2])/(2*dx)
    dudx[0] = (u[1] - u[0])/dx
    dudx[-1] = (u[-1] - u[-2])/dx
    
    # Update solution
    u = un - dt * (un * dudx)
    
    # Store solution
    u_sol[n+1] = u.copy()

# Plot results
plt.figure(figsize=(10,6))

# Initial condition
plt.plot(x, u_sol[0], 'b-', label='t=0')

# Solutions at different times
plt.plot(x, u_sol[int(nt/4)], 'g--', label=f't={T/4}')
plt.plot(x, u_sol[int(nt/2)], 'r-.', label=f't={T/2}')
plt.plot(x, u_sol[-1], 'k:', label=f't={T}')

plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Nonlinear Convection')
plt.legend()
plt.grid(True)
plt.show()