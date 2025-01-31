import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 41  # number of points in x direction
ny = 41  # number of points in y direction
length = 2.0  # domain length
height = 2.0  # domain height
rho = 1.0  # density
nu = 0.1  # kinematic viscosity
dt = 0.001  # time step
nit = 50  # number of pressure iterations
total_time = 3.0  # total simulation time

# Grid
dx = length / (nx - 1)
dy = height / (ny - 1)
x = np.linspace(0, length, nx)
y = np.linspace(0, height, ny)
X, Y = np.meshgrid(x, y)

# Initialize variables
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

def build_up_b(b, u, v, dx, dy, rho):
    """Build the source term for pressure equation"""
    b[1:-1, 1:-1] = rho * (
        (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
        (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
    ) / dt
    return b

def pressure_poisson(p, b, dx, dy):
    """Solve pressure Poisson equation"""
    pn = p.copy()
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
            (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2 -
            b[1:-1, 1:-1] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))
        
        # Boundary conditions for pressure
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        
    return p

def cavity_flow():
    t = 0
    u_n = np.zeros((ny, nx))
    v_n = np.zeros((ny, nx))
    
    while t < total_time:
        un = u.copy()
        vn = v.copy()
        
        # Build up b
        b = build_up_b(b, u, v, dx, dy, rho)
        
        # Solve pressure Poisson equation
        p = pressure_poisson(p, b, dx, dy)
        
        # Velocity update
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1] -
            un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
            vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
            dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
            nu * dt * (
                (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) / dx**2 +
                (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) / dy**2
            )
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1] -
            un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
            vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
            dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
            nu * dt * (
                (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) / dx**2 +
                (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) / dy**2
            )
        )
        
        # Boundary conditions
        u[-1, :] = 1    # u = 1 on top wall (lid)
        u[0, :] = 0     # u = 0 on bottom wall
        u[:, 0] = 0     # u = 0 on left wall
        u[:, -1] = 0    # u = 0 on right wall
        
        v[-1, :] = 0    # v = 0 on top wall
        v[0, :] = 0     # v = 0 on bottom wall
        v[:, 0] = 0     # v = 0 on left wall
        v[:, -1] = 0    # v = 0 on right wall
        
        t += dt

    return u, v, p

# Run simulation
u, v, p = cavity_flow()

# Plot results
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, u, levels=np.linspace(-0.1, 1, 20))
plt.colorbar(label='u-velocity')
plt.title('u-velocity field')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, v, levels=np.linspace(-0.5, 0.5, 20))
plt.colorbar(label='v-velocity')
plt.title('v-velocity field')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, p, levels=np.linspace(p.min(), p.max(), 20))
plt.colorbar(label='pressure')
plt.title('Pressure field')
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
    "v": v.tolist(),
    "p": p.tolist(),
    "u": u.tolist()
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
