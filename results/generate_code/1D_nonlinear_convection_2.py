import numpy as np
import matplotlib.pyplot as plt

# Grid setup
nx = 101
dx = 2.0/(nx-1)
x = np.linspace(0, 2, nx)

# Time setup 
nt = 500
dt = 2.0/nt

# Initial conditions
u = np.ones(nx)
u[int(0.5/dx):int(1/dx + 1)] = 2

# Initialize solution array to store results
un = np.ones(nx)

# Time stepping
for n in range(nt):
    un = u.copy()
    for i in range(1, nx-1):
        u[i] = un[i] - un[i] * dt/dx * (un[i] - un[i-1])
    
    # Boundary conditions
    u[0] = 1
    u[-1] = 1

# Plotting
plt.figure(figsize=(10,6))
plt.plot(x, u, 'b-', label='t=2')
plt.plot(x, np.ones(nx), 'k--', label='t=0')
plt.plot(x, [2 if 0.5<=xi<=1 else 1 for xi in x], 'r--', label='t=0')
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Nonlinear Convection')
plt.legend()
plt.grid(True)
plt.show()