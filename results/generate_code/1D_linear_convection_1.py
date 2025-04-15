import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
nx = 41          # Number of grid points
c = 1.0          # Wave speed
domain_length = 2.0  # Length of domain [0, 2]
nt = 50          # Number of timesteps
dt = 0.01        # Time step size

# Calculate dx and create spatial grid
dx = domain_length / (nx-1)
x = np.linspace(0, domain_length, nx)

# Set CFL number
CFL = c * dt / dx
print(f"CFL number: {CFL}")  # Should be less than 1 for stability

# Initialize solution array with initial conditions
u = np.ones(nx)  # First set all values to 1
# Set u = 2 for 0.5 <= x <= 1
u[(x >= 0.5) & (x <= 1)] = 2.0

# Store initial condition for comparison
u_initial = u.copy()

# Time stepping loop
for n in range(nt):
    # Copy u at previous timestep
    un = u.copy()
    
    # Update interior points using upwind scheme
    for i in range(1, nx):
        u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, u_initial, 'b-', label='Initial condition', linewidth=2)
plt.plot(x, u, 'r--', label=f'Solution at t = {nt*dt}', linewidth=2)
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Linear Convection')
plt.grid(True)
plt.legend()
plt.show()

# Create animation
from matplotlib.animation import FuncAnimation

# Reset initial conditions
u = np.ones(nx)
u[(x >= 0.5) & (x <= 1)] = 2.0

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, u)
ax.set_xlim(0, domain_length)
ax.set_ylim(0.5, 2.5)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.grid(True)

def animate(n):
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
    line.set_ydata(u)
    ax.set_title(f'Time = {n*dt:.2f}')
    return line,

anim = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
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
