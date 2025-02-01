import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
nu = 0.3  # Viscosity

# Discretization
Nx = 100  # Number of spatial points
Nt = 1000  # Number of time steps
dx = L / (Nx - 1)
dt = T / Nt

# Stability check (von Neumann analysis for diffusion equation)
stability_param = nu * dt / (dx * dx)
print(f"Stability parameter (should be ≤ 0.5): {stability_param}")

if stability_param > 0.5:
    print("Warning: Solution might be unstable!")

# Initialize arrays
x = np.linspace(0, L, Nx)
u = np.zeros((Nt + 1, Nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Time stepping (Beam-Warming scheme)
for n in range(0, Nt):
    for i in range(1, Nx-1):
        # Source term
        source = -np.pi**2 * nu * np.exp(-n*dt) * np.sin(np.pi * x[i]) + \
                 np.exp(-n*dt) * np.sin(np.pi * x[i])
        
        # Beam-Warming scheme for diffusion term
        u[n+1, i] = u[n, i] + dt * (
            nu * (u[n, i-1] - 2*u[n, i] + u[n, i+1]) / (dx**2) + 
            source
        )
    
    # Boundary conditions
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results at specific time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[Nt//4, :], 'r--', label=f't = {T/4}')
plt.plot(x, u[Nt//2, :], 'g-.', label=f't = {T/2}')
plt.plot(x, u[-1, :], 'k:', label=f't = {T}')

plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Diffusion Equation - Beam-Warming Scheme')
plt.legend()
plt.grid(True)
plt.show()

# Print maximum and minimum values
print(f"\nMaximum value: {np.max(u)}")
print(f"Minimum value: {np.min(u)}")
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
