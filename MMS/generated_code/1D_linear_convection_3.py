import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 80  # number of grid points
nt = 160  # number of time steps
dx = 2.0 / (nx - 1)  # grid spacing
dt = 2.0 / nt  # time step
c = 1.0  # wave speed
CFL = c * dt / dx  # Courant number

# Initialize arrays
x = np.linspace(0, 2, nx)
u = np.zeros((nt + 1, nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Source term function
def source(x, t):
    return -np.pi * c * np.exp(-t) * np.cos(np.pi * x) + np.exp(-t) * np.sin(np.pi * x)

# Beam-Warming scheme
for n in range(nt):
    un = u[n, :].copy()
    
    # Interior points using Beam-Warming scheme
    for i in range(2, nx-1):
        u[n+1, i] = (un[i] - 
                     CFL * (3*un[i] - 4*un[i-1] + un[i-2])/2 + 
                     dt * source(x[i], n*dt))
    
    # Boundary conditions
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results at key time steps
plt.figure(figsize=(10, 6))
key_times = [0, nt//4, nt//2, nt]
labels = ['t = 0', 't = T/4', 't = T/2', 't = T']
colors = ['b', 'g', 'r', 'k']

for i, t in enumerate(key_times):
    plt.plot(x, u[t, :], colors[i], label=labels[i])

plt.grid(True)
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Linear Convection - Beam-Warming Scheme')
plt.legend()
plt.show()

# Print stability condition
print(f"CFL number: {CFL}")
if CFL <= 1:
    print("Solution should be stable (CFL ≤ 1)")
else:
    print("Warning: Solution might be unstable (CFL > 1)")
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
