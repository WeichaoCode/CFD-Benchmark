import os
import json
import numpy as np
import matplotlib.pyplot as plt


def solve_1d_convection_fou():
    # Physical parameters
    c = 1.0  # Wave speed
    L = 2.0  # Domain length
    T = 2.0  # Total time

    # Numerical parameters
    nx = 100  # Number of spatial points
    dx = L / (nx - 1)  # Spatial step size

    # Apply CFL condition for stability (CFL ≤ 1)
    CFL = 0.8
    dt = CFL * dx / c
    nt = int(T / dt)  # Number of time steps

    # Initialize grid
    x = np.linspace(0, L, nx)
    u = np.sin(np.pi * x)  # Initial condition

    # Arrays to store solutions at key time steps
    u_t0 = u.copy()
    u_t1 = np.zeros_like(u)
    u_t2 = np.zeros_like(u)
    u_t3 = np.zeros_like(u)

    # Time stepping
    t = 0
    for n in range(nt):
        un = u.copy()

        # First Order Upwind scheme
        for i in range(1, nx - 1):
            source_term = -np.pi * c * np.exp(-t) * np.cos(np.pi * x[i]) + np.exp(-t) * np.sin(np.pi * x[i])
            u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1]) + dt * source_term

        # Apply boundary conditions
        u[0] = 0
        u[-1] = 0

        t += dt

        # Store solutions at key time steps
        if abs(t - T / 4) < dt:
            u_t1 = u.copy()
        elif abs(t - T / 2) < dt:
            u_t2 = u.copy()
        elif abs(t - T) < dt:
            u_t3 = u.copy()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_t0, 'b-', label='t = 0')
    plt.plot(x, u_t1, 'r--', label=f't = {T / 4:.2f}')
    plt.plot(x, u_t2, 'g-.', label=f't = {T / 2:.2f}')
    plt.plot(x, u_t3, 'm:', label=f't = {T:.2f}')

    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('1D Linear Convection - First Order Upwind')
    plt.legend()
    plt.grid(True)
    plt.show()
    return u_t3


# Run the simulation
u_t3 = solve_1d_convection_fou()
# Identify the filename of the running script
script_filename = os.path.basename(__file__)

# Define the JSON file
json_filename = "/opt/CFD-Benchmark/MMS/result/output.json"

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
    "u": u_t3.tolist()  # Convert NumPy array to list for JSON serialization
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
