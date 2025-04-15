import os
import json
import numpy as np
import matplotlib.pyplot as plt

def solve_laplace_2d():
    # Grid parameters
    Nx = 31  # Number of points in x direction
    Ny = 31  # Number of points in y direction
    
    # Domain boundaries
    x_min, x_max = 0, 2
    y_min, y_max = 0, 1
    
    # Grid spacing
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    
    # Create grid points
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialize solution array
    p = np.zeros((Ny, Nx))
    
    # Set boundary conditions
    # Right boundary: p = y at x = 2
    p[:, -1] = y
    
    # Iteration parameters
    max_iter = 10000
    tolerance = 1e-6
    error = 1.0
    
    # Gauss-Seidel iteration
    iteration = 0
    while error > tolerance and iteration < max_iter:
        p_old = p.copy()
        
        # Update interior points
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                # For boundary y = 0 and y = 1, implement Neumann condition
                if i == 0 or i == Ny-1:
                    p[i,j] = p[i+1,j] if i == 0 else p[i-1,j]
                else:
                    p[i,j] = 0.25 * (p[i+1,j] + p[i-1,j] + 
                                    p[i,j+1] + p[i,j-1])
        
        # Implement Neumann boundary conditions at y = 0 and y = 1
        p[0,:] = p[1,:]    # dp/dy = 0 at y = 0
        p[-1,:] = p[-2,:]  # dp/dy = 0 at y = 1
        
        # Calculate error
        error = np.max(np.abs(p - p_old))
        iteration += 1
    
    print(f"Solution converged after {iteration} iterations")
    return X, Y, p

def plot_solution(X, Y, p):
    # Create contour plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, p, levels=20, cmap='viridis')
    plt.colorbar(label='Pressure (p)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Laplace Equation Solution')
    
    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, p, cmap='viridis')
    fig.colorbar(surf, label='Pressure (p)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('p')
    ax.set_title('3D Surface Plot of Solution')
    plt.show()

# Solve the equation and plot results
X, Y, p = solve_laplace_2d()
plot_solution(X, Y, p)
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
    "p": p.tolist(),
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
