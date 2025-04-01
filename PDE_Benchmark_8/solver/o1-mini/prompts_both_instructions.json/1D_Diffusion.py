import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 41
nt = 20
nu = 0.3
dx = 2 / (nx - 1)
sigma = 0.2
dt = sigma * dx**2 / nu

# Spatial grid
x = np.linspace(0, 2, nx)

# Initial condition
u = np.ones(nx)
u[(x >= 0.5) & (x <= 1.0)] = 2.0

# Time-stepping
for _ in range(nt):
    un = u.copy()
    u[1:-1] = un[1:-1] + sigma * (un[2:] - 2 * un[1:-1] + un[:-2])
    u[0] = 1.0
    u[-1] = 0.0

# Visualization
plt.figure()
plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('u')
plt.title('Final solution at t=T')
plt.show()

# Save the final solution
np.save('u.npy', u)