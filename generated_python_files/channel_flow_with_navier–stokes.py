import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Domain and grid parameters
Lx, Ly = 2.0, 2.0  # Domain size
Nx, Ny = 41, 41    # Number of grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
dt = 0.01  # Time step size
nu = 0.1  # Kinematic viscosity
rho = 1.0  # Density
F = 1.0  # Source term (forcing)

# Create grids
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize fields
u = np.zeros((Ny, Nx))  # x-velocity
v = np.zeros((Ny, Nx))  # y-velocity
p = np.zeros((Ny, Nx))  # Pressure
b = np.zeros((Ny, Nx))  # RHS of Poisson equation

# Boundary conditions
def apply_boundary_conditions(u, v, p):
    # Periodic in x-direction
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]
    v[:, 0] = v[:, -2]
    v[:, -1] = v[:, 1]
    p[:, 0] = p[:, -2]
    p[:, -1] = p[:, 1]

    # No-slip on y-boundaries
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :] = 0

    # Pressure Neumann BC (∂p/∂y = 0)
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]

# Poisson equation solver for pressure correction
def pressure_poisson(p, b, tol=1e-5, max_iter=500):
    pn = np.copy(p)
    for _ in range(max_iter):
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                         (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2 -
                         b[1:-1, 1:-1] * dx**2 * dy**2 / rho) / (2 * (dx**2 + dy**2))
        apply_boundary_conditions(u, v, p)
        if np.linalg.norm(p - pn, ord=2) < tol:
            break
        pn = np.copy(p)
    return p

# Time-stepping loop
num_steps = 500  # Number of time steps
for n in range(num_steps):
    un = np.copy(u)
    vn = np.copy(v)

    # Compute RHS of pressure equation (divergence of velocity)
    b[1:-1, 1:-1] = (rho * ((un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx) +
                            (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)) / dt)

    # Solve Poisson equation for pressure correction
    p = pressure_poisson(p, b)

    # Update velocities using momentum equations
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - dt * (un[1:-1, 1:-1] * (un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx) +
                                            vn[1:-1, 1:-1] * (un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy)) -
                     dt * (1 / rho) * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx) +
                     nu * dt * ((un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) / dx**2 +
                                (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dy**2) + dt * F)

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - dt * (un[1:-1, 1:-1] * (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dx) +
                                            vn[1:-1, 1:-1] * (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)) -
                     dt * (1 / rho) * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy) +
                     nu * dt * ((vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) / dx**2 +
                                (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]) / dy**2))

    # Apply boundary conditions
    apply_boundary_conditions(u, v, p)

    # Print progress
    if n % 50 == 0:
        print(f"Step {n}/{num_steps} completed.")

# Visualization of velocity field
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, u, v)
plt.title("Velocity Field (u, v)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot pressure field
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, p, levels=50, cmap="coolwarm")
plt.colorbar(label="Pressure")
plt.title("Pressure Field (p)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Identify the filename of the running script
script_filename = os.path.basename(__file__)

# Define the JSON file
json_filename = "/opt/CFD-Benchmark/data/output_true.json"

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
    "p": p.tolist(),
    "u": u.tolist(),
    "v": v.tolist()
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")

    