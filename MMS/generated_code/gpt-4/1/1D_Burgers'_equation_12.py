import numpy as np
import matplotlib.pyplot as plt

# Constants
nu = 0.07
T = 2.0
L = 2.0
dt = 0.01
dx = 0.01
nt = int(T/dt)
nx = int(L/dx)
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
u = np.zeros((nt, nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Time stepping
for n in range(nt-1):
    for i in range(1, nx-1):
        u_xx = (u[n, i-1] - 2*u[n, i] + u[n, i+1]) / dx**2
        u_x = (u[n, i+1] - u[n, i]) / dx
        f = - u[n, i] * u_x + nu * u_xx - np.pi**2 * nu * np.exp(-t[n]) * np.sin(np.pi * x[i]) + np.exp(-t[n]) * np.sin(np.pi * x[i]) - np.pi * np.exp(-2*t[n]) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i])
        u[n+1, i] = u[n, i] + dt * f

# Plot the solution at key time steps
plt.figure(figsize=(8, 6))
for i, time in enumerate([0, nt//4, nt//2, nt-1]):
    plt.plot(x, u[time, :], label="t = " + str(time*dt))
plt.title("1D Burgers' equation solved by Beam-Warming method")
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()