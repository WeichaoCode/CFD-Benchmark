import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0/(nx-1)  # spatial step size
dt = 0.002  # time step size (chosen for stability)
c = 1.0  # wave speed

# Stability check (CFL condition for Lax-Friedrichs)
CFL = c*dt/dx
print(f"CFL number: {CFL}")
if CFL > 1:
    print("Warning: Solution might be unstable!")

# Initialize spatial and temporal grids
x = np.linspace(0, 2, nx)
t = np.linspace(0, 2, nt)

# Initialize solution array
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(np.pi*x)

# Source term function
def source(x, t):
    return -np.pi*c*np.exp(-t)*np.cos(np.pi*x) + np.exp(-t)*np.sin(np.pi*x)

# Lax-Friedrichs scheme
def lax_friedrichs_step(u_prev, dx, dt, c, t_current):
    u_next = np.zeros_like(u_prev)
    
    # Interior points
    for i in range(1, len(u_prev)-1):
        u_next[i] = 0.5*(u_prev[i+1] + u_prev[i-1]) - \
                    0.5*c*dt/dx*(u_prev[i+1] - u_prev[i-1]) + \
                    dt*source(x[i], t_current)
    
    # Boundary conditions
    u_next[0] = 0
    u_next[-1] = 0
    
    return u_next

# Time stepping
for n in range(nt-1):
    u[n+1, :] = lax_friedrichs_step(u[n, :], dx, dt, c, t[n])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[nt//4, :], 'r--', label=f't = {t[nt//4]:.2f}')
plt.plot(x, u[nt//2, :], 'g-.', label=f't = {t[nt//2]:.2f}')
plt.plot(x, u[-1, :], 'k:', label=f't = {t[-1]:.2f}')

plt.title('1D Linear Convection - Lax-Friedrichs Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Print maximum and minimum values for stability check
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
