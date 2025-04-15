import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha = 0.1  # Thermal diffusivity
Q0 = 200     # Maximum heating at center (0,0)
sigma = 0.1  # Controls radial decay of heat source
dx = 0.01    # space step
dt = 0.5 * dx**2 / alpha  # time step
T_cool = 0.0  # initial temperature

# Create the grid
x = np.arange(-1.0,1.0,dx)
y = np.arange(-1.0,1.0,dx)
T = np.full((len(x), len(y)), T_cool)  # Initial condition

# Time stepping function
def time_step(T):
    Tn = T.copy()
    T[1:-1, 1:-1] = Tn[1:-1, 1:-1] + alpha * dt / dx**2 * (Tn[2:, 1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1] + Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) + Q0 * np.exp(-(x[1:-1]**2 + y[1:-1]**2) / (2*sigma**2))
    # Enforce Neumann BCs
    T[:, -1] = T[:, -2]
    T[0, :] = T[1, :]
    T[:, 0] = T[:, 1]
    T[-1, :] = T[-2, :]
    return T

# Time-stepping
for n in range(1000):  # number of time steps
    T = time_step(T)

# Visualization
plt.imshow(T, cmap='hot', interpolation='nearest', extent=[-1, 1, -1, 1])
plt.colorbar(label="Temperature (°C)")
plt.title("Heat Equation")
plt.show()