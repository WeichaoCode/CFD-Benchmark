import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
Nx = 41  # Number of grid points in x
Ny = 41  # Number of grid points in y
Lx = 2.0  # Domain length in x
Ly = 2.0  # Domain length in y
dx = Lx / (Nx-1)  # Grid spacing in x
dy = Ly / (Ny-1)  # Grid spacing in y
rho = 1.0  # Density
nu = 0.1   # Kinematic viscosity
F = 1.0    # Source term
dt = 0.001  # Time step
nt = 1000   # Number of time steps

# Grid points
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize variables
u = np.zeros((Ny, Nx))  # x-velocity
v = np.zeros((Ny, Nx))  # y-velocity
p = np.zeros((Ny, Nx))  # pressure
un = np.zeros((Ny, Nx))  # temporary array for u
vn = np.zeros((Ny, Nx))  # temporary array for v
pn = np.zeros((Ny, Nx))  # temporary array for p

def central_diff_x(phi):
    return (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2*dx)

def central_diff_y(phi):
    return (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2*dy)

def laplacian(phi):
    d2x = (np.roll(phi, -1, axis=1) - 2*phi + np.roll(phi, 1, axis=1)) / dx**2
    d2y = (np.roll(phi, -1, axis=0) - 2*phi + np.roll(phi, 1, axis=0)) / dy**2
    return d2x + d2y

# Time stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Solve momentum equations
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                     dt * (un[1:-1, 1:-1] * central_diff_x(un)[1:-1, 1:-1] +
                          vn[1:-1, 1:-1] * central_diff_y(un)[1:-1, 1:-1]) +
                     dt * nu * laplacian(un)[1:-1, 1:-1] +
                     dt * F)
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     dt * (un[1:-1, 1:-1] * central_diff_x(vn)[1:-1, 1:-1] +
                          vn[1:-1, 1:-1] * central_diff_y(vn)[1:-1, 1:-1]) +
                     dt * nu * laplacian(vn)[1:-1, 1:-1])
    
    # Enforce boundary conditions
    # No-slip at y boundaries
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :] = 0
    
    # Periodic in x-direction
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]
    v[:, 0] = v[:, -2]
    v[:, -1] = v[:, 1]
    
    # Solve pressure Poisson equation
    for it in range(50):  # Iteration for pressure
        pn = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                         (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) / (2*(dx**2 + dy**2))
        
        # Periodic boundary conditions for pressure in x-direction
        p[:, 0] = p[:, -2]
        p[:, -1] = p[:, 1]
        
        # Neumann boundary condition for pressure in y-direction
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]

# Plot results
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.contourf(X, Y, u, levels=np.linspace(u.min(), u.max(), 20))
plt.colorbar(label='u-velocity')
plt.title('u-velocity')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(122)
plt.contourf(X, Y, v, levels=np.linspace(v.min(), v.max(), 20))
plt.colorbar(label='v-velocity')
plt.title('v-velocity')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# Plot velocity vectors
plt.figure(figsize=(8, 8))
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
plt.title('Velocity Field')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
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
    "v": v.tolist(),
    "p": p.tolist(),
    "u": u.tolist()
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
