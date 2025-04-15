import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Physical parameters
alpha = 0.25
Q_0 = 200
sigma = 0.1
L = 1.0  # Length of domain (-L, L)
T_0 = 0.0  # Boundary temperature

# Grid parameters
N = 101  # Number of grid points
dx = 2.0 * L / (N - 1)  # Grid spacing
dt = 0.5 * (dx*dx) / (4*alpha)  # Time step (CFL condition for stability)

# Create grid
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)

X, Y = np.meshgrid(x,y)

# Initial conditions
T = np.full((N, N), T_0)

# Source term
q = Q_0 * np.exp(- (X**2 + Y**2) / (2 * sigma**2))

# Create plot
fig, ax = plt.subplots(figsize=(5,5))
contour = ax.contourf(X, Y, T, 100)

# Time-stepping function
def update(i):
    global T
    T_new = T + dt * (alpha * (np.roll(T, -1, axis=0) - 2*T + np.roll(T, 1, axis=0)) / dx**2
                       + alpha * (np.roll(T, -1, axis=1) - 2*T + np.roll(T, 1, axis=1)) / dx**2
                       + q)
    # Enforce boundary conditions
    T_new[0, :] = T_new[-1, :] = T_new[:, 0] = T_new[:, -1] = T_0
    T = T_new
    ax.collections = []  # Clear previous plot
    contour = ax.contourf(X, Y, T, 100)
    return contour

# Create animation
ani = FuncAnimation(fig, update, frames=100, interval=100)
plt.show()