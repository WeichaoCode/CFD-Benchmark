import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0 / (nx - 1)  # spatial step size
dt = 2.0 / nt  # time step size
x = np.linspace(0, 2, nx)  # spatial grid
t = np.linspace(0, 2, nt)  # time grid

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

# Beam-Warming scheme
def beam_warming_step(u_prev, dt, dx, t_current):
    u_new = u_prev.copy()
    
    # Interior points
    for i in range(2, nx-1):
        # Beam-Warming discretization for nonlinear convection term
        if u_prev[i] >= 0:
            convection = u_prev[i] * (3*u_prev[i] - 4*u_prev[i-1] + u_prev[i-2])/(2*dx)
        else:
            convection = u_prev[i] * (-u_prev[i+2] + 4*u_prev[i+1] - 3*u_prev[i])/(2*dx)
            
        # Update solution
        u_new[i] = u_prev[i] - dt * convection + dt * source(x[i], t_current)
    
    return u_new

# Time integration
for n in range(nt-1):
    u[n+1] = beam_warming_step(u[n], dt, dx, t[n])
    u[n+1, 0] = 0  # Enforce boundary conditions
    u[n+1, -1] = 0

# Plot results at key time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4)], 'r--', label=f't = {t[int(nt/4)]:.2f}')
plt.plot(x, u[int(nt/2)], 'g-.', label=f't = {t[int(nt/2)]:.2f}')
plt.plot(x, u[-1], 'k:', label=f't = {t[-1]:.2f}')

plt.title('1D Nonlinear Convection - Beam-Warming Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Print maximum absolute value to check stability
print(f"Maximum absolute value: {np.max(np.abs(u))}")
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
