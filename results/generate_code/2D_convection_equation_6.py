import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Define parameters
Nx = 101  # Number of points in x-direction
Ny = 101  # Number of points in y-direction
Lx = 2.0  # Length of domain in x-direction
Ly = 2.0  # Length of domain in y-direction
dx = Lx / (Nx-1)  # Grid spacing in x
dy = Ly / (Ny-1)  # Grid spacing in y
dt = 0.001  # Time step
t_final = 1.0  # Final time
Nt = int(t_final/dt)  # Number of time steps

# Create spatial grids
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize velocity fields
u = np.ones((Ny, Nx))
v = np.ones((Ny, Nx))

# Set initial conditions
for i in range(Ny):
    for j in range(Nx):
        if 0.5 <= x[j] <= 1.0 and 0.5 <= y[i] <= 1.0:
            u[i,j] = 2.0
            v[i,j] = 2.0

# Create arrays to store new values
u_new = np.zeros((Ny, Nx))
v_new = np.zeros((Ny, Nx))

def apply_boundary_conditions(u, v):
    # Apply boundary conditions
    u[0,:] = 1.0  # Bottom boundary
    u[-1,:] = 1.0  # Top boundary
    u[:,0] = 1.0  # Left boundary
    u[:,-1] = 1.0  # Right boundary
    
    v[0,:] = 1.0  # Bottom boundary
    v[-1,:] = 1.0  # Top boundary
    v[:,0] = 1.0  # Left boundary
    v[:,-1] = 1.0  # Right boundary
    return u, v

# Time stepping loop
for n in range(Nt):
    # Apply upwind scheme for interior points
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            # x-derivatives
            if u[i,j] > 0:
                du_dx = (u[i,j] - u[i,j-1])/dx
                dv_dx = (v[i,j] - v[i,j-1])/dx
            else:
                du_dx = (u[i,j+1] - u[i,j])/dx
                dv_dx = (v[i,j+1] - v[i,j])/dx
            
            # y-derivatives
            if v[i,j] > 0:
                du_dy = (u[i,j] - u[i-1,j])/dy
                dv_dy = (v[i,j] - v[i-1,j])/dy
            else:
                du_dy = (u[i+1,j] - u[i,j])/dy
                dv_dy = (v[i+1,j] - v[i,j])/dy
            
            # Update velocities
            u_new[i,j] = u[i,j] - dt*(u[i,j]*du_dx + v[i,j]*du_dy)
            v_new[i,j] = v[i,j] - dt*(u[i,j]*dv_dx + v[i,j]*dv_dy)
    
    # Update values and apply boundary conditions
    u = u_new.copy()
    v = v_new.copy()
    u, v = apply_boundary_conditions(u, v)

    # Print progress
    if n % 100 == 0:
        print(f"Time step {n}/{Nt}")

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.contourf(X, Y, u, levels=50, cmap=cm.viridis)
plt.colorbar(label='u velocity')
plt.title('u-velocity field')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(122)
plt.contourf(X, Y, v, levels=50, cmap=cm.viridis)
plt.colorbar(label='v velocity')
plt.title('v-velocity field')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# Plot velocity vectors
plt.figure(figsize=(8, 6))
skip = 5  # Plot every 5th vector for clarity
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
          u[::skip, ::skip], v[::skip, ::skip])
plt.title('Velocity Vector Field')
plt.xlabel('x')
plt.ylabel('y')
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
    "v": v.tolist(),
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
