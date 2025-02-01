import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0 / (nx - 1)  # spatial step size
dt = 2.0 / (nt - 1)  # time step size

# Initialize arrays
x = np.linspace(0, 2, nx)
u = np.sin(np.pi * x)  # initial condition
u_old = u.copy()

# For stability, check CFL condition
# For nonlinear equation, we use max|u| for stability analysis
# CFL = max|u|*dt/dx should be less than 1
CFL = np.max(np.abs(u)) * dt / dx
print(f"CFL number: {CFL}")
if CFL >= 1:
    print("Warning: Solution might be unstable!")

# Time steps for plotting
plot_times = [0, 0.5, 1.0, 2.0]  # T/4, T/2, T
plot_indices = [int(t/dt) for t in plot_times]
solutions_to_plot = {0: u.copy()}

# Time stepping
t = 0
for n in range(1, nt):
    t = n * dt
    u_old = u.copy()
    
    # First order upwind scheme
    for i in range(1, nx-1):
        # Source terms
        source = np.exp(-t) * np.sin(np.pi * x[i]) - \
                np.pi * np.exp(-2*t) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i])
        
        # Upwind scheme based on sign of u
        if u_old[i] >= 0:
            u[i] = u_old[i] - dt * (u_old[i] * (u_old[i] - u_old[i-1]) / dx) + dt * source
        else:
            u[i] = u_old[i] - dt * (u_old[i] * (u_old[i+1] - u_old[i]) / dx) + dt * source
    
    # Apply boundary conditions
    u[0] = 0
    u[-1] = 0
    
    # Store solutions at plot times
    if n in plot_indices:
        solutions_to_plot[t] = u.copy()

# Plotting
plt.figure(figsize=(10, 6))
for t, solution in solutions_to_plot.items():
    plt.plot(x, solution, label=f't = {t:.1f}')

plt.title('1D Nonlinear Convection - First Order Upwind')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
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
    "u": solutions_to_plot[-1].tolist()  # Convert NumPy array to list for JSON serialization
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
