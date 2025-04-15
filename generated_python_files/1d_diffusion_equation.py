import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Physical properties
nu = 0.3  # viscosity

# Domain and discretization
Lx = 2.0  # domain length
Nx = 41   # number of grid points
dx = Lx / (Nx - 1)  # grid spacing

# Time parameters
sigma = 0.2  # CFL-like number for stability
dt = sigma * dx**2 / nu  # time step
Nt = 100  # number of time steps

# Discretized space
x = np.linspace(0, Lx, Nx)

# Initial condition
u = np.ones(Nx)  # Initialize with 1
u[(x >= 0.5) & (x <= 1.0)] = 2  # Apply condition

# Time-stepping loop using explicit finite difference
for n in range(Nt):
    u_new = u.copy()  # Copy current state for update
    for i in range(1, Nx-1):  # Exclude boundaries
        u_new[i] = u[i] + nu * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])

    u = u_new.copy()  # Update solution

# Plot results
plt.plot(x, u, label=f"t = {Nt * dt:.2f}")
plt.xlabel("x")
plt.ylabel("u")
plt.title("1D Diffusion Equation Solution")
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
