import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
L = 2.0   # length of domain
T = 2.0   # total time
dx = L / (nx-1)
dt = T / nt
nu = 0.07  # viscosity

# Stability check (von Neumann analysis)
c = dt/dx
if c > 1:
    print("Warning: CFL condition not satisfied!")

# Grid points
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Initialize solution array
u = np.zeros((nt, nx))

# Initial condition
u[0,:] = np.sin(np.pi*x)

# Source term function
def source(x, t):
    return (-np.pi**2 * nu * np.exp(-t) * np.sin(np.pi*x) + 
            np.exp(-t) * np.sin(np.pi*x) - 
            np.pi * np.exp(-2*t) * np.sin(np.pi*x) * np.cos(np.pi*x))

# Lax-Friedrichs scheme
def lax_friedrichs_step(u_prev, dx, dt, nu):
    u_new = np.zeros_like(u_prev)
    
    # Interior points
    for i in range(1, nx-1):
        # Convective term
        conv = -0.25*(u_prev[i+1]**2 - u_prev[i-1]**2)/dx
        
        # Diffusive term
        diff = nu*(u_prev[i+1] - 2*u_prev[i] + u_prev[i-1])/(dx**2)
        
        # Source term
        src = source(x[i], t[j])
        
        # Lax-Friedrichs update
        u_new[i] = 0.5*(u_prev[i+1] + u_prev[i-1]) + dt*(conv + diff + src)
    
    # Boundary conditions
    u_new[0] = 0
    u_new[-1] = 0
    
    return u_new

# Time stepping
for j in range(nt-1):
    u[j+1] = lax_friedrichs_step(u[j], dx, dt, nu)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, u[0], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4)], 'r--', label=f't = {T/4:.2f}')
plt.plot(x, u[int(nt/2)], 'g-.', label=f't = {T/2:.2f}')
plt.plot(x, u[-1], 'k:', label=f't = {T:.2f}')

plt.title("1D Burgers' Equation - Lax-Friedrichs Method")
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()

# Print maximum values at different times for stability check
print(f"Max value at t=0: {np.max(np.abs(u[0])):.4f}")
print(f"Max value at t=T/4: {np.max(np.abs(u[int(nt/4)])):.4f}")
print(f"Max value at t=T/2: {np.max(np.abs(u[int(nt/2)])):.4f}")
print(f"Max value at t=T: {np.max(np.abs(u[-1])):.4f}")
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
