import math
import numpy as np
import matplotlib.pyplot as plt

# Set the parameters
nx = 101
nt = 100
T = 2.0
dt = T / (nt - 1)
c = 1.0
dx = 2.0 / (nx - 1)

# Initialize the solution array
u = np.zeros((nt, nx))

# Set the initial condition
for i in range(nx):
    u[0, i] = math.sin(math.pi * i * dx)

# Solve the equation
for n in range(nt - 1):
    for i in range(1, nx - 1):
        u[n + 1, i] = u[n, i] - dt / dx * u[n, i] * (u[n, i] - u[n, i - 1]) + dt * math.exp(-n * dt) * math.sin(math.pi * i * dx) - dt * math.pi * math.exp(-2 * n * dt) * math.sin(math.pi * i * dx) * math.cos(math.pi * i * dx)

# Plot the solution
plt.figure(figsize=(6, 4))
plt.plot(np.linspace(0, 2, nx), u[0, :], label='t = 0')
plt.plot(np.linspace(0, 2, nx), u[nt // 4, :], label='t = T/4')
plt.plot(np.linspace(0, 2, nx), u[nt // 2, :], label='t = T/2')
plt.plot(np.linspace(0, 2, nx), u[-1, :], label='t = T')
plt.legend()
plt.title('1D Nonlinear Convection - Finite Difference Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.show()