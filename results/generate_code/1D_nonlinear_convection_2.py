import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nx = 41  # Number of spatial points
dx = 2.0/(nx-1)  # Spatial step size
nt = 25  # Number of time steps
dt = 0.025  # Time step size
c = 1  # CFL number (for stability)

# Initialize domain
x = np.linspace(0, 2, nx)  # Create spatial grid
u = np.ones(nx)  # Initialize solution array with ones

# Set initial conditions (IC)
# u = 2 for 0.5 <= x <= 1, and u = 1 elsewhere
u[(x >= 0.5) & (x <= 1)] = 2.0

# Store initial condition for plotting
u_initial = u.copy()

# Solve the equation
for n in range(nt):
    # Create a copy of u for the previous time step
    un = u.copy()
    
    # Apply finite difference scheme
    # Forward difference in time, backward difference in space
    # u[i]^(n+1) = u[i]^n - u[i]^n * dt/dx * (u[i]^n - u[i-1]^n)
    for i in range(1, nx):
        u[i] = un[i] - un[i] * dt/dx * (un[i] - un[i-1])

    # Apply boundary condition at x = 0
    u[0] = 1.0

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, u_initial, 'b-', label='Initial Condition', linewidth=2)
plt.plot(x, u, 'r--', label='Final Solution', linewidth=2)
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Nonlinear Convection')
plt.legend()
plt.grid(True)
plt.show()

# Create animation of the solution
from matplotlib import animation

# Initialize animation setup
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(xlim=(0, 2), ylim=(0.5, 2.5))
line, = ax.plot([], [], 'b-', linewidth=2)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Nonlinear Convection Wave Propagation')

# Initialize data
u = np.ones(nx)
u[(x >= 0.5) & (x <= 1)] = 2.0

# Animation initialization function
def init():
    line.set_data([], [])
    return line,

# Animation function
def animate(n):
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - un[i] * dt/dx * (un[i] - un[i-1])
    u[0] = 1.0
    
    line.set_data(x, u)
    return line,

# Create animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=nt, interval=100, blit=True)
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
