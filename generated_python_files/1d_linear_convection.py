import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Physical parameters
c = 1  # Wave speed
Nx = 41  # Number of spatial grid points
Lx = 2.0  # Domain length
dx = Lx / (Nx - 1)  # Grid spacing
dt = 0.025  # Time step (should satisfy CFL condition)
Nt = 25  # Number of time steps

# Create spatial grid
x = np.linspace(0, Lx, Nx)

# Initialize u with the given initial condition
u = np.ones(Nx)  # u = 1 everywhere
u[(x >= 0.5) & (x <= 1)] = 2  # Set u = 2 in the range 0.5 ≤ x ≤ 1

# Plot initial condition
plt.plot(x, u, label="Initial Condition", color='k')

# Time integration using Upwind Scheme
for n in range(Nt):
    u_new = u.copy()  # Copy current u for update
    for i in range(1, Nx):  # Upwind scheme requires i-1 index
        u_new[i] = u[i] - c * dt / dx * (u[i] - u[i-1])
    u = u_new.copy()  # Update solution

# Plot final solution
plt.plot(x, u, label="Final Condition (t={:.2f})".format(Nt * dt), linestyle='--', color='r')
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Linear Convection Equation')
plt.legend()
plt.grid()
plt.show()

# Identify the filename of the running script
script_filename = os.path.basename(__file__)

# Define the JSON file
json_filename = "/opt/CFD-Benchmark/data/output_generate.json"

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

