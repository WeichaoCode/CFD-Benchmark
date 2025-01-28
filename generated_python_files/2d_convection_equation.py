import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Define parameters
Lx, Ly = 2.0, 2.0  # Domain size
Nx, Ny = 101, 101  # Number of grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
sigma = 0.2  # CFL-like stability parameter
dt = sigma * min(dx, dy)  # Time step
Nt = 100  # Number of time steps

# Create mesh grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize velocity field
u = np.ones((Ny, Nx))  # u velocity field
v = np.ones((Ny, Nx))  # v velocity field

# Apply initial condition
u[(X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)] = 2
v[(X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)] = 2

# Time-stepping loop using upwind scheme
for n in range(Nt):
    u_new = u.copy()
    v_new = v.copy()

    # Compute upwind differences for u
    u_new[1:, 1:] = u[1:, 1:] - dt / dx * u[1:, 1:] * (u[1:, 1:] - u[1:, :-1])                                 - dt / dy * v[1:, 1:] * (u[1:, 1:] - u[:-1, 1:])

    # Compute upwind differences for v
    v_new[1:, 1:] = v[1:, 1:] - dt / dx * u[1:, 1:] * (v[1:, 1:] - v[1:, :-1])                                 - dt / dy * v[1:, 1:] * (v[1:, 1:] - v[:-1, 1:])

    # Apply boundary conditions
    u_new[0, :], u_new[:, 0] = 1, 1  # Left and bottom boundaries
    u_new[-1, :], u_new[:, -1] = 1, 1  # Right and top boundaries

    v_new[0, :], v_new[:, 0] = 1, 1  # Left and bottom boundaries
    v_new[-1, :], v_new[:, -1] = 1, 1  # Right and top boundaries

    # Update the solution
    u, v = u_new.copy(), v_new.copy()

# Plot results
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, u, cmap="coolwarm")
plt.colorbar(label="u(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Convection Equation Solution (u)")
plt.show()

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, v, cmap="coolwarm")
plt.colorbar(label="v(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Convection Equation Solution (v)")
plt.show()

# Identify the filename of the running script
script_filename = os.path.basename(__file__)

# Define the JSON file
json_filename = "/opt/CFD-Benchmark/data/output_generate.json"

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

