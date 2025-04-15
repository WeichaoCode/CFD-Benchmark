import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Define parameters
nu = 0.01  # Viscosity
Lx, Ly = 2.0, 2.0  # Domain size
Nx, Ny = 41, 41  # Number of grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
# sigma = 0.2
sigma = 0.0009  # CFL-like stability parameter chatgpt used to choose 0.2 which cannot ensure stable
# dt = sigma * min(dx, dy) ** 2 / nu  # Time step
dt = sigma * dx * dy / nu
Nt = 500  # Number of time steps

# Create mesh grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize velocity fields
u = np.ones((Ny, Nx))  # x-direction velocity
v = np.ones((Ny, Nx))  # y-direction velocity

# Apply initial condition
u[(X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)] = 2
v[(X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)] = 2

# Time-stepping loop using explicit finite difference
for n in range(Nt):
    u_new = u.copy()
    v_new = v.copy()

    # Compute upwind differences for convection and central differences for diffusion
    u_new[1:-1, 1:-1] = (
        u[1:-1, 1:-1]
        - dt / dx * u[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[1:-1, :-2])  # Convective term (du/dx)
        - dt / dy * v[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[:-2, 1:-1])  # Convective term (du/dy)
        + nu * dt / dx**2 * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])  # Diffusive term (d²u/dx²)
        + nu * dt / dy**2 * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])  # Diffusive term (d²u/dy²)
    )

    v_new[1:-1, 1:-1] = (
        v[1:-1, 1:-1]
        - dt / dx * u[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[1:-1, :-2])  # Convective term (dv/dx)
        - dt / dy * v[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[:-2, 1:-1])  # Convective term (dv/dy)
        + nu * dt / dx**2 * (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2])  # Diffusive term (d²v/dx²)
        + nu * dt / dy**2 * (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1])  # Diffusive term (d²v/dy²)
    )

    # Apply boundary conditions (Dirichlet)
    u_new[0, :], u_new[:, 0] = 1, 1  # Left and bottom boundaries
    u_new[-1, :], u_new[:, -1] = 1, 1  # Right and top boundaries

    v_new[0, :], v_new[:, 0] = 1, 1  # Left and bottom boundaries
    v_new[-1, :], v_new[:, -1] = 1, 1  # Right and top boundaries

    # Update solution
    u, v = u_new.copy(), v_new.copy()

# Plot results
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, u, cmap="coolwarm")
plt.colorbar(label="u(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Burgers' Equation Solution (u)")
plt.show()

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, v, cmap="coolwarm")
plt.colorbar(label="v(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Burgers' Equation Solution (v)")
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


    