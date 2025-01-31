import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
Nx = 41  # Number of grid points in x
Ny = 41  # Number of grid points in y
Lx = 2.0  # Domain length in x
Ly = 2.0  # Domain length in y
dx = Lx / (Nx-1)  # Grid spacing in x
dy = Ly / (Ny-1)  # Grid spacing in y
nu = 0.01  # Viscosity
dt = 0.001  # Time step
t_final = 1.0  # Final time
Nt = int(t_final/dt)  # Number of time steps

# Grid points
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize velocity fields
u = np.ones((Ny, Nx))
v = np.ones((Ny, Nx))

# Set initial conditions
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2.0
v[mask] = 2.0

# Function to apply boundary conditions
def apply_bc(u, v):
    # Apply boundary conditions (u = v = 1 at all boundaries)
    u[0, :] = 1.0  # Bottom
    u[-1, :] = 1.0  # Top
    u[:, 0] = 1.0  # Left
    u[:, -1] = 1.0  # Right
    
    v[0, :] = 1.0  # Bottom
    v[-1, :] = 1.0  # Top
    v[:, 0] = 1.0  # Left
    v[:, -1] = 1.0  # Right
    return u, v

# Time stepping using finite differences
def time_step(u, v):
    # Create copies for the new time step
    un = u.copy()
    vn = v.copy()
    
    # Interior points
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            # Central differences for spatial derivatives
            ux = (u[i,j+1] - u[i,j-1])/(2*dx)
            uy = (u[i+1,j] - u[i-1,j])/(2*dy)
            uxx = (u[i,j+1] - 2*u[i,j] + u[i,j-1])/dx**2
            uyy = (u[i+1,j] - 2*u[i,j] + u[i-1,j])/dy**2
            
            vx = (v[i,j+1] - v[i,j-1])/(2*dx)
            vy = (v[i+1,j] - v[i-1,j])/(2*dy)
            vxx = (v[i,j+1] - 2*v[i,j] + v[i,j-1])/dx**2
            vyy = (v[i+1,j] - 2*v[i,j] + v[i-1,j])/dy**2
            
            # Update u and v
            un[i,j] = u[i,j] + dt*(nu*(uxx + uyy) - u[i,j]*ux - v[i,j]*uy)
            vn[i,j] = v[i,j] + dt*(nu*(vxx + vyy) - u[i,j]*vx - v[i,j]*vy)
    
    return un, vn

# Main time loop
for n in range(Nt):
    # Update u and v
    u, v = time_step(u, v)
    
    # Apply boundary conditions
    u, v = apply_bc(u, v)
    
    if n % 100 == 0:
        print(f"Time step {n}/{Nt}")

# Plotting the results
fig = plt.figure(figsize=(15, 5))

# Plot u
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, u, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')
ax1.set_title('u-velocity')
plt.colorbar(surf1, ax=ax1)

# Plot v
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, v, cmap='viridis')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('v')
ax2.set_title('v-velocity')
plt.colorbar(surf2, ax=ax2)

plt.tight_layout()
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
    "v": v.tolist()
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
