import numpy as np
import matplotlib.pyplot as plt

# Define the simulation parameters
c = 1.0
x_start = 0.0
x_end = 2.0
t_start = 0.0
t_end = 2.0
nx = 101
nt = 1001
dx = (x_end - x_start) / (nx - 1)
dt = (t_end - t_start) / (nt - 1)
x = np.linspace(x_start, x_end, nx)
t = np.linspace(t_start, t_end, nt)

# Initialize the solution array
u = np.zeros((nt, nx))

# Set the initial condition
u[0, :] = np.sin(np.pi * x)

# Set the boundary conditions
u[:, 0] = 0.0
u[:, -1] = 0.0

# Time-stepping loop
for n in range(nt - 1):
    for i in range(1, nx - 1):
        # FTCS scheme
        u[n + 1, i] = u[n, i] - c * dt / dx * (u[n, i + 1] - u[n, i - 1]) / 2 \
        + dt * np.pi * c * np.exp(-t[n]) * np.cos(np.pi * x[i]) \
        - dt * np.exp(-t[n]) * np.sin(np.pi * x[i])

# Plot the solution at key time steps
plt.figure(figsize=(10, 8))
plt.plot(x, u[0, :], label="t = 0")
plt.plot(x, u[nt//4, :], label="t = T/4")
plt.plot(x, u[nt//2, :], label="t = T/2")
plt.plot(x, u[-1, :], label="t = T")
plt.legend()
plt.title("1D Linear Convection - FTCS method")
plt.xlabel("x")
plt.ylabel("u")
plt.show()