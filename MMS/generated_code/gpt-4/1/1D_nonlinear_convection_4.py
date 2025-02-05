import numpy as np
import matplotlib.pyplot as plt

# Define the problem parameters
T = 2.0
L = 2.0
nx = 100
nt = 100
dx = L / (nx - 1)
dt = T / (nt - 1)
x = np.linspace(0, L, nx)

# Initialize solution matrix
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Apply boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Solve using FOU
for t in range(nt - 1):
    for i in range(1, nx - 1):
        u[t + 1, i] = u[t, i] - dt / dx * u[t, i] * (u[t, i] - u[t, i - 1]) + dt * (np.exp(-t*dt) * np.sin(np.pi * x[i]) - np.pi * np.exp(-2*t*dt) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i]))

# Plot the solution at key time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], label='t = 0')
plt.plot(x, u[int(nt / 4), :], label='t = T/4')
plt.plot(x, u[int(nt / 2), :], label='t = T/2')
plt.plot(x, u[-1, :], label='t = T')
plt.legend()
plt.title('1D Nonlinear Convection - First Order Upwind')
plt.xlabel('x')
plt.ylabel('u')
plt.grid()
plt.show()