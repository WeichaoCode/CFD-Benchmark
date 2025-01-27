import json
import os

# Directory to save Python files
code_dir = "generated_python_files"
os.makedirs(code_dir, exist_ok=True)

# JSON file to store problem-solution mapping
output_json_path = "generated_cfd_solutions.json"

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
    "name": "1D Burgers' equation",
    "prompt": "Solve the 1D Burgers' equation problem using Python...",
    "generated_code": """import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nu = 0.07  # Viscosity
Lx = 2 * np.pi  # Domain length
Nx = 101  # Number of spatial points
dx = Lx / (Nx - 1)  # Grid spacing
sigma = 0.2  # CFL-like parameter for stability
dt = sigma * dx**2 / nu  # Time step
Nt = 200  # Number of time steps

# Define spatial grid
x = np.linspace(0, Lx, Nx)

# Define initial condition
phi = np.exp(-x**2 / (4 * nu)) + np.exp(-(x - 2*np.pi)**2 / (4 * nu))
dphi_dx = (-x / (4 * nu) * np.exp(-x**2 / (4 * nu))) + (-(x - 2*np.pi) / (4 * nu) * np.exp(-(x - 2*np.pi)**2 / (4 * nu)))
u = -2 * nu * dphi_dx / phi + 4  # Compute initial velocity

# Time stepping loop
for n in range(Nt):
    u_new = u.copy()  # Copy current state for update
    
    # First derivative (convection term) using upwind scheme
    du_dx = (u - np.roll(u, 1)) / dx
    
    # Second derivative (diffusion term) using central difference
    d2u_dx2 = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
    
    # Update equation (Explicit scheme)
    u_new = u - dt * u * du_dx + nu * dt * d2u_dx2

    # Enforce periodic boundary conditions
    u = u_new.copy()

# Plot results
plt.plot(x, u, label=f"t = {Nt * dt:.2f}")
plt.xlabel("x")
plt.ylabel("u")
plt.title("1D Burgers' Equation Solution")
plt.legend()
plt.grid()
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

