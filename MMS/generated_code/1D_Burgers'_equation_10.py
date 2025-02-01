import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
nu = 0.07  # Viscosity
nx = 100  # Number of spatial points
nt = 1000  # Number of time steps
dx = L / (nx - 1)
dt = T / nt
x = np.linspace(0, L, nx)

# Stability check (CFL condition)
c = dt / dx
d = nu * dt / (dx * dx)
print(f"CFL number: {c}")
print(f"Diffusion number: {d}")

# Initialize solution array
u = np.zeros((nt + 1, nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Time stepping
for n in range(nt):
    # Internal points
    for i in range(1, nx-1):
        # First order upwind for convection term
        if u[n, i] >= 0:
            conv = u[n, i] * (u[n, i] - u[n, i-1]) / dx
        else:
            conv = u[n, i] * (u[n, i+1] - u[n, i]) / dx
            
        # Central difference for diffusion term
        diff = nu * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / (dx*dx)
        
        # Source terms
        source = (-np.pi**2 * nu * np.exp(-n*dt) * np.sin(np.pi*x[i]) + 
                 np.exp(-n*dt) * np.sin(np.pi*x[i]) - 
                 np.pi * np.exp(-2*n*dt) * np.sin(np.pi*x[i]) * np.cos(np.pi*x[i]))
        
        # Update solution
        u[n+1, i] = u[n, i] - dt * conv + dt * diff + dt * source

    # Apply boundary conditions
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results at key time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4), :], 'r--', label=f't = {T/4}')
plt.plot(x, u[int(nt/2), :], 'g-.', label=f't = {T/2}')
plt.plot(x, u[-1, :], 'k:', label=f't = {T}')
plt.xlabel('x')
plt.ylabel('u')
plt.title("1D Burgers' Equation - First Order Upwind")
plt.legend()
plt.grid(True)
plt.show()
# Identify the filename of the running script
script_filename = os.path.basename(__file__)

# Define the JSON file
json_filename = "/opt/CFD-Benchmark/MMS/result/output.json"

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
    "u": u[-1, :].tolist()  # Convert NumPy array to list for JSON serialization
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
