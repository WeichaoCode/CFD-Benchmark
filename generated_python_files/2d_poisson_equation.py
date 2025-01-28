import numpy as np
import matplotlib.pyplot as plt
import os
import json

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
    "p": p.tolist()
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")

    