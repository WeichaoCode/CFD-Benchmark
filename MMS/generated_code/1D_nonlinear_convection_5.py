import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # Number of spatial points
nt = 1000  # Number of time steps
dx = 2.0 / (nx - 1)  # Spatial step size
dt = 2.0 / (nt - 1)  # Time step size
x = np.linspace(0, 2, nx)  # Spatial domain
t = np.linspace(0, 2, nt)  # Temporal domain

# Initialize solution array
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Set boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Source term function
def source(x, t):
    return np.exp(-t) * np.sin(np.pi * x) - np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x)

# Lax-Friedrichs scheme
def lax_friedrichs_step(u_prev, dt, dx, t):
    u_next = np.zeros_like(u_prev)
    
    # Apply Lax-Friedrichs scheme for interior points
    for i in range(1, len(u_prev)-1):
        # Flux term
        flux_term = 0.5 * (u_prev[i+1]**2 - u_prev[i-1]**2) / (2*dx)
        
        # Diffusion term from Lax-Friedrichs
        diff_term = (u_prev[i+1] + u_prev[i-1]) / 2 - u_prev[i]
        
        # Source term
        src = source(x[i], t)
        
        u_next[i] = (u_prev[i+1] + u_prev[i-1])/2 - dt * flux_term + dt * src
    
    # Apply boundary conditions
    u_next[0] = 0
    u_next[-1] = 0
    
    return u_next

# Time stepping
for n in range(nt-1):
    u[n+1] = lax_friedrichs_step(u[n], dt, dx, t[n])

# Plot results at specified time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4)], 'r--', label=f't = {t[int(nt/4)]:.2f}')
plt.plot(x, u[int(nt/2)], 'g-.', label=f't = {t[int(nt/2)]:.2f}')
plt.plot(x, u[-1], 'k:', label=f't = {t[-1]:.2f}')

plt.title('1D Nonlinear Convection - Lax-Friedrichs Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()

# Print CFL number for stability check
c = np.max(np.abs(u)) * dt/dx
print(f"CFL number: {c:.3f}")
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
    "u": u[-1].tolist()  # Convert NumPy array to list for JSON serialization
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
