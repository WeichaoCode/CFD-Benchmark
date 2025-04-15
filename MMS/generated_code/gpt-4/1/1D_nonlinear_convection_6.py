import numpy as np
import matplotlib.pyplot as plt

# Define the initial conditions
x_start, x_end = 0, 2
t_start, t_end = 0, 2
n_x, n_t = 100, 100

dx = (x_end - x_start) / (n_x - 1)
dt = (t_end - t_start) / (n_t - 1)

x = np.linspace(x_start, x_end, n_x)
t = np.linspace(t_start, t_end, n_t)

u = np.zeros((n_t, n_x))

# Set the boundary conditions
u[:, 0] = u[:, -1] = 0

# Set the initial condition
u[0, :] = np.sin(np.pi * x)

# Define the nonlinear convection function
def convection(u, x, t):
    return - u * (u[1:] - u[:-1]) / dx - np.exp(-t) * np.sin(np.pi * x[1:]) + np.pi * np.exp(-2 * t) * np.sin(np.pi * x[1:]) * np.cos(np.pi * x[1:])

# Solve the equation using the Beam-Warming method
for n in range(n_t - 1):
    u[n + 1, 2:] = u[n, 2:] + dt * convection(u[n, :], x[2:], t[n])

# Plot the solution
plt.figure(figsize=(10, 6))
plt.title('1D Nonlinear Convection using Beam-Warming Method')

for i, time in enumerate([0, n_t // 4, n_t // 2, n_t - 1]):
    plt.plot(x, u[time, :], label=f't = {time * dt}')

plt.legend()
plt.grid(True)
plt.show()