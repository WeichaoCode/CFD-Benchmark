import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
nu = 0.3  # Viscosity

# Grid parameters
Nx = 100  # Number of spatial points
Nt = 1000  # Number of time points
dx = L / (Nx - 1)
dt = T / (Nt - 1)

# Stability check (von Neumann analysis)
r = nu * dt / (dx * dx)
print(f"r = {r}")
if r > 0.5:
    print("Warning: Scheme might be unstable!")

# Grid points
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Initialize solution array
u = np.zeros((Nt, Nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Time stepping using Beam-Warming scheme
for n in range(0, Nt-1):
    for i in range(1, Nx-1):
        # Source term
        source = -np.pi**2 * nu * np.exp(-t[n]) * np.sin(np.pi * x[i]) + \
                 np.exp(-t[n]) * np.sin(np.pi * x[i])
        
        # Beam-Warming scheme
        u[n+1, i] = u[n, i] + \
                    nu * dt/(dx**2) * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) + \
                    dt * source

    # Boundary conditions
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results at specific time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[int(Nt/4), :], 'r--', label=f't = {T/4:.2f}')
plt.plot(x, u[int(Nt/2), :], 'g-.', label=f't = {T/2:.2f}')
plt.plot(x, u[-1, :], 'k:', label=f't = {T:.2f}')

plt.title('1D Diffusion Equation - Beam-Warming Scheme')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.grid(True)
plt.legend()
plt.show()

# Print max and min values
print(f"Maximum value: {np.max(u):.6f}")
print(f"Minimum value: {np.min(u):.6f}")