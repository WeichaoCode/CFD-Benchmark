import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Parameters
nu = 0.07  # viscosity
L = 2*np.pi  # domain length
Nx = 101  # number of spatial points
dx = L/(Nx-1)  # spatial step size
dt = 0.001  # time step
t_end = 2.0  # end time
Nt = int(t_end/dt)  # number of time steps

# Spatial grid
x = np.linspace(0, L, Nx)

def initial_condition(x, nu):
    """Define the initial condition"""
    phi = np.exp(-(x**2)/(4*nu)) + np.exp(-((x - 2*np.pi)**2)/(4*nu))
    phi_x = (-x/(2*nu))*np.exp(-(x**2)/(4*nu)) + \
            (-(x - 2*np.pi)/(2*nu))*np.exp(-((x - 2*np.pi)**2)/(4*nu))
    return -2*nu/phi * phi_x + 4

def solve_burgers():
    # Initialize solution array
    u = np.zeros((Nt+1, Nx))
    u[0,:] = initial_condition(x, nu)
    
    # Time stepping
    for n in range(Nt):
        # Periodic boundary conditions
        u[n,0] = u[n,-2]
        u[n,-1] = u[n,1]
        
        # Spatial derivatives
        u_x = (u[n,2:] - u[n,:-2])/(2*dx)
        u_xx = (u[n,2:] - 2*u[n,1:-1] + u[n,:-2])/(dx**2)
        
        # Update solution (excluding boundary points)
        u[n+1,1:-1] = u[n,1:-1] - dt*u[n,1:-1]*u_x + nu*dt*u_xx
        
        # Update periodic boundaries
        u[n+1,0] = u[n+1,-2]
        u[n+1,-1] = u[n+1,1]
    
    return u

# Solve the equation
u = solve_burgers()

# Create animation
fig, ax = plt.subplots()
line, = ax.plot(x, u[0,:])
ax.set_xlim(0, L)
ax.set_ylim(np.min(u)-0.5, np.max(u)+0.5)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('1D Burgers Equation')

def animate(frame):
    line.set_ydata(u[frame,:])
    return line,

anim = animation.FuncAnimation(fig, animate, frames=len(u), 
                             interval=50, blit=True)

# Plot final solution
plt.figure()
plt.plot(x, u[-1,:], 'b-', label='Final solution')
plt.plot(x, u[0,:], 'r--', label='Initial condition')
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Burgers Equation: Initial and Final Solutions')
plt.legend()
plt.grid(True)
plt.show()
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
    "u": u[-1, :].tolist(),
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
