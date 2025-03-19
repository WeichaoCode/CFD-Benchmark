import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the parameters
nx, ny = 41, 41
t_max = 3.0
alpha = 1.0
sigma = 0.1
Q0 = 200.0
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
beta = dx / dy
r = 0.5 / (1 + beta**2)
dt = r * dx**2 / alpha

# Initialize the solution arrays
T = np.zeros((nx, ny, 2), dtype=np.float64)
T_new = np.zeros((nx, ny), dtype=np.float64)

# Define the source term
def q(x, y):
    return Q0 * np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Time-stepping loop
for n in range(int(t_max / dt)):
    # Update the solution at the interior points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            T_new[i, j] = (2*r*(T[i+1, j, 1] + T[i-1, j, 1]) + 2*(beta**2)*r*(T[i, j+1, 1] + T[i, j-1, 1]) + T[i, j, 0] + 2*dt*q(x[i], y[j])) / (1 + 2*r + 2*(beta**2)*r)
    # Update the solution at the boundary points
    T_new[0, :], T_new[-1, :], T_new[:, 0], T_new[:, -1] = 0, 0, 0, 0
    # Update the solution arrays for the next time step
    T[:, :, 0], T[:, :, 1] = T[:, :, 1], T_new

# Save the solution in .npy format
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/T_2D_Unsteady_Heat_Equation_DF.npy', T_new)

# Visualize the temperature evolution
fig, ax = plt.subplots()
cax = ax.imshow(T[:, :, 1], cmap='hot', origin='lower')
bar = fig.colorbar(cax)

def update(frame):
    ax.set_title(f'Time: {frame*dt:.2f} s')
    cax.set_array(T[:, :, frame % 2].flatten())

ani = FuncAnimation(fig, update, frames=range(int(t_max / dt)), interval=100)
plt.show()