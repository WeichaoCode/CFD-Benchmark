import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def solve_2d_diffusion():
    # Physical parameters
    nu = 0.05  # viscosity
    
    # Domain parameters
    x_start, x_end = 0, 2
    y_start, y_end = 0, 2
    Nx = Ny = 31  # number of grid points
    
    # Grid parameters
    dx = (x_end - x_start) / (Nx - 1)
    dy = (y_end - y_start) / (Ny - 1)
    x = np.linspace(x_start, x_end, Nx)
    y = np.linspace(y_start, y_end, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Time parameters
    dt = 0.25 * min(dx, dy)**2 / nu  # stability condition
    t_end = 1.0
    nt = int(t_end/dt)
    
    # Initialize solution array
    u = np.ones((Ny, Nx))
    
    # Set initial conditions
    for i in range(Nx):
        for j in range(Ny):
            if (0.5 <= x[i] <= 1.0) and (0.5 <= y[j] <= 1.0):
                u[j,i] = 2.0
    
    # Store initial condition
    u_initial = u.copy()
    
    # Time stepping
    for n in range(nt):
        u_old = u.copy()
        
        # Interior points
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                u[j,i] = u_old[j,i] + nu*dt*(
                    (u_old[j,i+1] - 2*u_old[j,i] + u_old[j,i-1])/dx**2 +
                    (u_old[j+1,i] - 2*u_old[j,i] + u_old[j-1,i])/dy**2
                )
        
        # Boundary conditions
        u[0,:] = 1  # bottom
        u[-1,:] = 1  # top
        u[:,0] = 1  # left
        u[:,-1] = 1  # right
    
    return X, Y, u_initial, u

def plot_results(X, Y, u_initial, u_final):
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    
    # Plot initial condition
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_initial, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax1.set_title('Initial Condition')
    fig.colorbar(surf1, ax=ax1)
    
    # Plot final solution
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_final, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    ax2.set_title('Final Solution')
    fig.colorbar(surf2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

# Solve the equation and plot results
X, Y, u_initial, u_final = solve_2d_diffusion()
plot_results(X, Y, u_initial, u_final)
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
    "u": u_final.tolist()
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
