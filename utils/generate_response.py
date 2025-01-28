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
    "name": "2D Poisson equation",
    "prompt": "Solve the 2D Poisson equation problem using Python...",
    "generated_code": """import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Lx, Ly = 2.0, 1.0  # Domain size
Nx, Ny = 50, 50  # Number of grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
tolerance = 1e-4  # Convergence criterion
max_iterations = 5000  # Maximum number of iterations
omega = 1.5  # Over-relaxation factor for SOR (optional)

# Create mesh grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize solution and source term
p = np.zeros((Ny, Nx))  # Solution array
b = np.zeros((Ny, Nx))  # Source term array

# Apply source term
b[int(Ny / 4), int(Nx / 4)] = 100
b[int(3 * Ny / 4), int(3 * Nx / 4)] = -100

# Iterative solver using Gauss-Seidel
error = 1.0
iteration = 0

while error > tolerance and iteration < max_iterations:
    p_old = p.copy()
    
    # Finite difference update for Poisson equation
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            p[i, j] = (1 - omega) * p_old[i, j] + omega * 0.25 * (
                p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1] - dx**2 * b[i, j]
            )
    
    # Apply boundary conditions (Dirichlet: p = 0)
    p[0, :], p[-1, :], p[:, 0], p[:, -1] = 0, 0, 0, 0

    # Compute error for convergence check
    error = np.max(np.abs(p - p_old))
    iteration += 1

print(f"Converged in {iteration} iterations with error {error:.6f}")

# Plot results
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, p, cmap="coolwarm", levels=50)
plt.colorbar(label="p(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Poisson Equation Solution")
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

