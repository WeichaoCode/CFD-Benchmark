import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # Number of spatial points
nt = 1000  # Number of time steps
dx = 2.0 / (nx - 1)  # Spatial step size
dt = 2.0 / (nt - 1)  # Time step size
nu = 0.07  # Viscosity

# Initialize arrays
x = np.linspace(0, 2, nx)
u = np.zeros((nt, nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Stability check (von Neumann analysis)
CFL = dt / dx
if CFL > 1:
    print("Warning: Solution might be unstable! CFL =", CFL)

def source_term(x, t):
    """Calculate the source term"""
    return (-np.pi**2 * nu * np.exp(-t) * np.sin(np.pi * x) + 
            np.exp(-t) * np.sin(np.pi * x) - 
            np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x))

# Beam-Warming scheme
for n in range(0, nt-1):
    t = n * dt
    
    # Apply boundary conditions
    u[n, 0] = 0
    u[n, -1] = 0
    
    # Interior points
    for i in range(2, nx-1):
        # Beam-Warming for convective term
        convective = (u[n,i] * (3*u[n,i] - 4*u[n,i-1] + u[n,i-2]) / (2*dx))
        
        # Central difference for diffusive term
        diffusive = nu * (u[n,i+1] - 2*u[n,i] + u[n,i-1]) / (dx**2)
        
        # Source term
        source = source_term(x[i], t)
        
        # Update solution
        u[n+1,i] = u[n,i] - dt * convective + dt * diffusive + dt * source

    # Apply boundary conditions again
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, u[0,:], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4),:], 'r--', label=f't = {0.5:.1f}')
plt.plot(x, u[int(nt/2),:], 'g-.', label=f't = {1.0:.1f}')
plt.plot(x, u[-1,:], 'k:', label=f't = {2.0:.1f}')

plt.title("1D Burgers' Equation - Beam-Warming Scheme")
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Print maximum values at different times
print(f"Max value at t=0: {np.max(u[0,:]):.4f}")
print(f"Max value at t=T/4: {np.max(u[int(nt/4),:]):.4f}")
print(f"Max value at t=T/2: {np.max(u[int(nt/2),:]):.4f}")
print(f"Max value at t=T: {np.max(u[-1,:]):.4f}")
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
