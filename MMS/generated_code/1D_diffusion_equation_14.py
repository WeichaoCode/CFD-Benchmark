import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
nu = 0.3  # Viscosity

# Grid parameters
Nx = 100  # Number of spatial points
Nt = 1000  # Number of time steps
dx = L / (Nx - 1)
dt = T / Nt

# Stability check (von Neumann analysis)
stability_parameter = nu * dt / (dx * dx)
print(f"Stability parameter (should be ≤ 0.5): {stability_parameter}")

if stability_parameter > 0.5:
    print("Warning: Solution might be unstable!")

# Initialize grid
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Initialize solution array
u = np.zeros((Nt, Nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Time-stepping solution
def source_term(x, t):
    return -np.pi**2 * nu * np.exp(-t) * np.sin(np.pi * x) + np.exp(-t) * np.sin(np.pi * x)

# Solve using First Order Upwind scheme
for n in range(0, Nt-1):
    for i in range(1, Nx-1):
        # Central difference for second derivative
        d2u_dx2 = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / (dx**2)
        u[n+1, i] = u[n, i] + dt * (nu * d2u_dx2 + source_term(x[i], t[n]))
    
    # Apply boundary conditions
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results
plt.figure(figsize=(10, 6))
time_steps = [0, Nt//4, Nt//2, Nt-1]
labels = ['t = 0', f't = T/4', f't = T/2', f't = T']
colors = ['b', 'g', 'r', 'k']

for i, n in enumerate(time_steps):
    plt.plot(x, u[n, :], colors[i], label=labels[i])

plt.title('1D Diffusion Equation - First Order Upwind')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.grid(True)
plt.legend()
plt.show()

# Calculate and print maximum value for each time step
for i, n in enumerate(time_steps):
    print(f"Maximum value at {labels[i]}: {np.max(np.abs(u[n, :])):.6f}")