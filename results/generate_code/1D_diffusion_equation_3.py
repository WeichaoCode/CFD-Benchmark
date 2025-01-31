import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0          # Domain length
Nx = 41          # Number of spatial points
dx = L/(Nx-1)    # Spatial step size
nu = 0.3         # Diffusion coefficient (viscosity)

# Time parameters
dt = 0.0001      # Time step (chosen small enough for stability)
t_final = 0.5    # Final time
Nt = int(t_final/dt)  # Number of time steps

# Stability check (CFL condition)
stability_param = nu*dt/(dx**2)
print(f"Stability parameter (should be < 0.5): {stability_param}")

# Initialize spatial grid
x = np.linspace(0, L, Nx)

# Initialize solution array
u = np.ones(Nx)  # Initialize with u=1 everywhere

# Set initial condition: u=2 for 0.5<=x<=1, u=1 elsewhere
u[(x >= 0.5) & (x <= 1.0)] = 2.0

# Create copy for new time step
u_new = u.copy()

# Time stepping
for n in range(Nt):
    # Interior points
    for i in range(1, Nx-1):
        u_new[i] = u[i] + nu*dt/(dx**2) * (u[i+1] - 2*u[i] + u[i-1])
    
    # Boundary conditions (zero gradient)
    u_new[0] = u_new[1]      # Left boundary
    u_new[-1] = u_new[-2]    # Right boundary
    
    # Update solution
    u = u_new.copy()

    # Plot at specific time intervals
    if n % 1000 == 0:
        plt.plot(x, u, label=f't = {n*dt:.3f}')

# Plotting
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Diffusion Equation Solution')
plt.grid(True)
plt.legend()
plt.show()

# Final solution plot
plt.figure()
plt.plot(x, u, 'b-', label='Final solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title(f'1D Diffusion Equation Solution at t = {t_final}')
plt.grid(True)
plt.legend()
plt.show()
##############################################
# The following lines are used to print output
##############################################

# Identify the filename of the running script
script_filename = os.path.basename(__file__)

# Define the JSON file
json_filename = "/opt/CFD-Benchmark/results/output_pred.json"

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
    "u": u.tolist(),
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
