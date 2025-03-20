import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define Parameters
nx, ny = 41, 41
nt = 120
x_start, x_end = 0, 2
y_start, y_end = 0, 2
nu = 0.01
sigma = 0.0009

# Computed Parameters
dx = (x_end - x_start) / (nx - 1)
dy = (y_end - y_start) / (ny - 1)
dt = sigma * dx * dy / nu

# Initialize Variables
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Apply Initial Condition
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
x_grid, y_grid = np.meshgrid(x, y)

# Set u, v to 2 in the region 0.5 <= x <= 1 and 0.5 <= y <= 1
u[np.logical_and(np.logical_and(x_grid >= 0.5, x_grid <= 1),
                 np.logical_and(y_grid >= 0.5, y_grid <= 1))] = 2
v[np.logical_and(np.logical_and(x_grid >= 0.5, x_grid <= 1),
                 np.logical_and(y_grid >= 0.5, y_grid <= 1))] = 2

# Time Integration Loop
for n in range(nt):
    u_temp = u.copy()
    v_temp = v.copy()

    # Update u and v using Explicit Euler Method & Central Differences
    u[1:-1, 1:-1] = (u_temp[1:-1, 1:-1] - 
                     dt / dx * u_temp[1:-1, 1:-1] * (u_temp[1:-1, 1:-1] - u_temp[1:-1, :-2]) -
                     dt / dy * v_temp[1:-1, 1:-1] * (u_temp[1:-1, 1:-1] - u_temp[:-2, 1:-1]) +
                     nu * dt / dx**2 * (u_temp[1:-1, 2:] - 2 * u_temp[1:-1, 1:-1] + u_temp[1:-1, :-2]) +
                     nu * dt / dy**2 * (u_temp[2:, 1:-1] - 2 * u_temp[1:-1, 1:-1] + u_temp[:-2, 1:-1]))

    v[1:-1, 1:-1] = (v_temp[1:-1, 1:-1] - 
                     dt / dx * u_temp[1:-1, 1:-1] * (v_temp[1:-1, 1:-1] - v_temp[1:-1, :-2]) -
                     dt / dy * v_temp[1:-1, 1:-1] * (v_temp[1:-1, 1:-1] - v_temp[:-2, 1:-1]) +
                     nu * dt / dx**2 * (v_temp[1:-1, 2:] - 2 * v_temp[1:-1, 1:-1] + v_temp[1:-1, :-2]) +
                     nu * dt / dy**2 * (v_temp[2:, 1:-1] - 2 * v_temp[1:-1, 1:-1] + v_temp[:-2, 1:-1]))

    # Apply Boundary Conditions
    u[0, :], u[-1, :], u[:, 0], u[:, -1] = 1, 1, 1, 1
    v[0, :], v[-1, :], v[:, 0], v[:, -1] = 1, 1, 1, 1

# Save final fields
np.save('u_final.npy', u)
np.save('v_final.npy', v)

# Visualization Function
def plot_field(X, Y, field, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, field, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Velocity')
    plt.show()

# Plotting the fields
plot_field(x_grid, y_grid, u, '/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/u_2D_Burgers_Equation.npy')
plot_field(x_grid, y_grid, v, '/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/v_2D_Burgers_Equation.npy')