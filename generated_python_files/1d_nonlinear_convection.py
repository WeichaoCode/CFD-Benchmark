import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Define parameters
Nx = 41  
Lx = 2.0  
dx = Lx / (Nx - 1)  
dt = 0.0125  
Nt = 20  

# Create spatial grid
x = np.linspace(0, Lx, Nx)

# Initialize u
u = np.ones(Nx)
u[(x >= 0.5) & (x <= 1)] = 2  

# Time integration using Upwind Scheme
for n in range(Nt):
    u_new = u.copy()
    for i in range(1, Nx):
        u_new[i] = u[i] - u[i] * dt / dx * (u[i] - u[i-1])
    u = u_new.copy()

# Plot result
plt.plot(x, u)
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
