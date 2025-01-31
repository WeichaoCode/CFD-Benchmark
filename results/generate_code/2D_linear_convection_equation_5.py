import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameters
nx = 81  # number of grid points in x
ny = 81  # number of grid points in y
c = 1.0  # wave speed
x_length = 2.0  # domain length in x
y_length = 2.0  # domain length in y
t_final = 1.0  # final time

# Calculate grid parameters
dx = x_length / (nx - 1)
dy = y_length / (ny - 1)
dt = 0.4 * min(dx, dy) / c  # CFL condition
nt = int(t_final / dt)  # number of time steps

# Create grid
x = np.linspace(0, x_length, nx)
y = np.linspace(0, y_length, ny)
X, Y = np.meshgrid(x, y)

# Initialize solution array
u = np.ones((ny, nx))  # initialize with ones

# Set initial conditions
# u = 2 for 0.5 ≤ x, y ≤ 1
u[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2.0

# Initialize array for next time step
un = u.copy()

# Time stepping
for n in range(nt):
    un = u.copy()
    
    # Update interior points using upwind scheme
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                     c * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     c * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]))
    
    # Boundary conditions
    u[0, :] = 1.0  # bottom boundary
    u[-1, :] = 1.0  # top boundary
    u[:, 0] = 1.0  # left boundary
    u[:, -1] = 1.0  # right boundary

# Plotting
def plot_solution(X, Y, u, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, u, cmap=cm.viridis)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u')
    ax.set_title(title)
    plt.colorbar(surf)
    plt.show()

# Plot initial condition
plot_solution(X, Y, un, 'Initial Condition')

# Plot final solution
plot_solution(X, Y, u, f'Solution at t = {t_final}')

# Plot contour
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, u, levels=20, cmap='viridis')
plt.colorbar(label='u')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Contour plot at t = {t_final}')
plt.show()
##############################################
# The following lines are used to print output
##############################################

# Identify the filename of the running script
script_filename = os.path.basename(__file__)

# Define the JSON file
json_filename = "/opt/CFD-Benchmark/results/output_pred.json"

# Load existing JSON data if the file exists
if os.path.exists(json_filename):
    with open(json_filename, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = {}  # Handle empty or corrupted file
else:
    data = {}

# Save filename and output array in a structured format
data[script_filename] = {
    "filename": script_filename,
    "u": u.tolist()
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
