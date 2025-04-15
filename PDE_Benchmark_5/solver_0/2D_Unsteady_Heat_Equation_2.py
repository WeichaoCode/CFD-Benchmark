import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Define a mesh
nx, ny = 100, 100
dx, dy = 2.0 / (nx - 1), 2.0 / (ny - 1)
X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))

# Define system properties
alpha = 1e-1
dt = (dx * dy) / (4 * alpha)
t_end = 1.0

# Initialize the field
T = np.zeros_like(X)
T_next = np.zeros_like(T)

def evolve(T, T_next, dt, dx, dy, alpha):
    # Implement Forward difference for time and Central difference for space
    T_next[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * dt / dx**2 * (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1])\
                                        + alpha * dt / dy**2 * (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2])

t = 0.0
while t < t_end:
    t += dt
    evolve(T, T_next, dt, dx, dy, alpha)
    T, T_next = T_next, T

# Display the result
plt.imshow(T, cmap='hot', extent=(-1, 1, -1, 1))
plt.show()