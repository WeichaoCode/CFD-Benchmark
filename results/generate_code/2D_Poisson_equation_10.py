import os
import json
import numpy as np
import matplotlib.pyplot as plt

def solve_poisson_2d():
    # Grid parameters
    nx = 50  # Number of points in x-direction
    ny = 50  # Number of points in y-direction
    
    # Domain dimensions
    Lx = 2.0  # Length in x-direction
    Ly = 1.0  # Length in y-direction
    
    # Grid spacing
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    # Create grid points
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialize pressure field
    p = np.zeros((ny, nx))
    
    # Initialize source term
    b = np.zeros((ny, nx))
    # Set source/sink terms
    b[ny//4, nx//4] = 100
    b[3*ny//4, 3*nx//4] = -100
    
    # Iteration parameters
    max_iter = 10000
    tolerance = 1e-4
    
    # Gauss-Seidel iteration
    for it in range(max_iter):
        p_old = p.copy()
        
        # Update interior points
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                p[i,j] = ((p[i+1,j] + p[i-1,j])/dx**2 +
                         (p[i,j+1] + p[i,j-1])/dy**2 -
                         b[i,j]) / (2/dx**2 + 2/dy**2)
        
        # Apply boundary conditions
        p[0,:] = 0  # Bottom boundary
        p[-1,:] = 0  # Top boundary
        p[:,0] = 0  # Left boundary
        p[:,-1] = 0  # Right boundary
        
        # Check convergence
        error = np.max(np.abs(p - p_old))
        if error < tolerance:
            print(f"Solution converged after {it+1} iterations")
            break
            
    return X, Y, p

def plot_solution(X, Y, p):
    plt.figure(figsize=(10, 5))
    
    # Plot pressure contours
    plt.subplot(121)
    plt.contourf(X, Y, p, levels=50, cmap='RdBu')
    plt.colorbar(label='Pressure')
    plt.title('Pressure Contours')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Plot surface
    ax = plt.subplot(122, projection='3d')
    surf = ax.plot_surface(X, Y, p, cmap='RdBu')
    plt.colorbar(surf, label='Pressure')
    ax.set_title('Pressure Surface')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('p')
    
    plt.tight_layout()
    plt.show()

# Solve the equation and plot results
X, Y, p = solve_poisson_2d()
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
    "p": p.tolist()
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
