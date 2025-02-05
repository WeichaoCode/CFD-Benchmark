import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 1
pi = np.pi
x_start = 0
x_end = 2
t_start = 0
t_end = 2
nx = 101
nt = 101
dx = (x_end - x_start) / (nx - 1)
dt = (t_end - t_start) / (nt - 1)

# Stability condition
assert dt <= dx**2 / np.max(np.abs(c)), "Solution is not stable"

# Initialize the solution grid
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(pi * np.linspace(x_start, x_end, nx))

# Set boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Solve the PDE
for t in range(nt-1):
    F = c * np.roll(u[t, :], -1) - pi * c * np.exp(-t*dt) * np.cos(pi * np.linspace(x_start, x_end, nx)) + np.exp(-t*dt) * np.sin(pi * np.linspace(x_start, x_end, nx))
    u[t+1, 1:-1] = 0.5 * (u[t, :-2] + u[t, 2:]) - dt / (2*dx) * (F[2:] - F[:-2])

# Plot the solution
plt.figure(figsize=(10, 6))
for i, t in enumerate([0, nt//4, nt//2, nt-1]):
    plt.plot(np.linspace(x_start, x_end, nx), u[t, :], label=f"t = {t*dt}")
plt.title("1D Linear Convection - Lax-Friedrichs Method")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
plt.show()