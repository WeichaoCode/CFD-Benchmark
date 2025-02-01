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
dx = L / (Nx-1)
dt = T / Nt

# Stability check (von Neumann analysis for diffusion equation)
stability_param = nu * dt / (dx**2)
print(f"Stability parameter: {stability_param}")
if stability_param > 0.5:
    print("Warning: Solution might be unstable!")

# Initialize grid
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Initialize solution array
u = np.zeros((Nt, Nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Set boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Lax-Friedrichs method
def lax_friedrichs_step(u_prev, dx, dt, nu, t):
    u_new = np.zeros_like(u_prev)
    
    # Apply Lax-Friedrichs scheme for interior points
    for i in range(1, len(u_prev)-1):
        # Diffusion term
        diff_term = nu * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1]) / (dx**2)
        # Source terms
        source = -np.pi**2 * nu * np.exp(-t) * np.sin(np.pi * x[i]) + np.exp(-t) * np.sin(np.pi * x[i])
        
        u_new[i] = 0.5 * (u_prev[i+1] + u_prev[i-1]) + dt * (diff_term + source)
    
    return u_new

# Time stepping
for n in range(Nt-1):
    u[n+1, :] = lax_friedrichs_step(u[n, :], dx, dt, nu, t[n])
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results at key time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[int(Nt/4), :], 'r--', label=f't = {T/4:.1f}')
plt.plot(x, u[int(Nt/2), :], 'g-.', label=f't = {T/2:.1f}')
plt.plot(x, u[-1, :], 'k:', label=f't = {T:.1f}')

plt.title('1D Diffusion Equation - Lax-Friedrichs Method')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True)
plt.show()

# Print maximum and minimum values
print(f"Maximum value: {np.max(u)}")
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
