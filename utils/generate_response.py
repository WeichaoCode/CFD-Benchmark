import json
import os

# Directory to save Python files
code_dir = "../generated_python_files"
os.makedirs(code_dir, exist_ok=True)

# JSON file to store problem-solution mapping
output_json_path = "../data/generated_cfd_solutions.json"

# Load existing JSON data if the file exists
if os.path.exists(output_json_path):
    with open(output_json_path, "r") as json_file:
        try:
            json_data = json.load(json_file)
        except json.JSONDecodeError:
            json_data = {"solutions": []}  # Initialize if JSON is empty
else:
    json_data = {"solutions": []}

# Manually add solutions (copy-paste GPT-generated code here)
new_solution = {
    "name": "Cavity Flow with Navier–Stokes",
    "prompt": "Solve the Cavity Flow with Navier–Stokes problem using Python...",
    "generated_code": """import numpy as np
import matplotlib.pyplot as plt

# Domain and grid parameters
Lx, Ly = 2.0, 2.0  # Domain size
Nx, Ny = 41, 41    # Number of grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
dt = 0.001  # Time step size
nu = 0.1  # Kinematic viscosity
rho = 1.0  # Density

# Create grids
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize fields
u = np.zeros((Ny, Nx))  # x-velocity
v = np.zeros((Ny, Nx))  # y-velocity
p = np.zeros((Ny, Nx))  # Pressure
b = np.zeros((Ny, Nx))  # RHS of Poisson equation

# Apply boundary conditions
def apply_boundary_conditions(u, v, p):
    # Lid-driven boundary (top wall, y = Ly)
    u[-1, :] = 1  # Moving lid velocity
    v[-1, :] = 0

    # No-slip conditions at other walls
    u[0, :] = 0   # Bottom wall
    v[0, :] = 0
    u[:, 0] = 0   # Left wall
    v[:, 0] = 0
    u[:, -1] = 0  # Right wall
    v[:, -1] = 0

    # Pressure boundary conditions (Neumann ∂p/∂y = 0 at y = 0)
    p[0, :] = p[1, :]
    p[-1, :] = 0  # Dirichlet p = 0 at the lid
    p[:, 0] = p[:, 1]  # Neumann at x = 0
    p[:, -1] = p[:, -2]  # Neumann at x = Lx

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
                                (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dy**2))

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

    """
}

# Generate a filename based on the problem name
file_name = new_solution["name"].replace(" ", "_").lower() + ".py"
file_path = os.path.join(code_dir, file_name)

# Save the generated code to a Python file
with open(file_path, "w") as py_file:
    py_file.write(new_solution["generated_code"])

# Check if the solution already exists in JSON and replace it
found = False
for solution in json_data["solutions"]:
    if solution["name"] == new_solution["name"]:
        solution["prompt"] = new_solution["prompt"]
        solution["file_name"] = file_name
        found = True
        break

# If not found, append as a new entry
if not found:
    json_data["solutions"].append({
        "name": new_solution["name"],
        "prompt": new_solution["prompt"],
        "file_name": file_name
    })

# Save updated JSON file
with open(output_json_path, "w") as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"Solution saved for '{new_solution['name']}' in {file_path}")
print(f"Updated JSON saved to {output_json_path}")

