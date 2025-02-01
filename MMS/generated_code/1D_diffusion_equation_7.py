import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
nu = 0.3  # Viscosity

# Grid parameters
Nx = 100  # Number of spatial points
Nt = 1000  # Number of time steps
dx = L / (Nx - 1)
dt = T / Nt

# Ensure stability (von Neumann analysis for diffusion equation)
stability_parameter = nu * dt / (dx ** 2)
print(f"Stability parameter (should be ≤ 0.5): {stability_parameter}")

# Initialize grid
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Initialize solution array
u = np.zeros((Nt, Nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Time-stepping solution
for n in range(0, Nt-1):
    for i in range(1, Nx-1):
        # Finite difference approximation
        d2u_dx2 = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / (dx**2)
        source = -np.pi**2 * nu * np.exp(-t[n]) * np.sin(np.pi * x[i]) + np.exp(-t[n]) * np.sin(np.pi * x[i])
        
        # Update solution
        u[n+1, i] = u[n, i] + dt * (nu * d2u_dx2 + source)
    
    # Apply boundary conditions
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results at key time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[int(Nt/4), :], 'r--', label=f't = {T/4:.1f}')
plt.plot(x, u[int(Nt/2), :], 'g-.', label=f't = {T/2:.1f}')
plt.plot(x, u[-1, :], 'k:', label=f't = {T:.1f}')

plt.title('1D Diffusion Equation - First Order Upwind')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.grid(True)
plt.legend()
plt.show()

# Print maximum value of solution for verification
print(f"Maximum value of solution: {np.max(u)}")
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
