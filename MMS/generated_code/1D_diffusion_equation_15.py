import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
nu = 0.3  # Viscosity

# Discretization
Nx = 100  # Number of spatial points
Nt = 1000  # Number of time steps
dx = L / (Nx - 1)
dt = T / Nt

# Stability check (von Neumann analysis for Leapfrog)
CFL = nu * dt / (dx * dx)
print(f"CFL number: {CFL}")
if CFL >= 0.5:
    print("Warning: Scheme might be unstable!")

# Grid points
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Initialize solution array
u = np.zeros((Nt, Nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# First time step using Forward Euler (needed for Leapfrog)
for i in range(1, Nx-1):
    u[1, i] = u[0, i] + dt * (
        nu * (u[0, i+1] - 2*u[0, i] + u[0, i-1])/(dx*dx) 
        - np.pi**2 * nu * np.exp(-t[0]) * np.sin(np.pi*x[i])
        + np.exp(-t[0]) * np.sin(np.pi*x[i])
    )

# Boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Time stepping using Leapfrog
for n in range(1, Nt-1):
    for i in range(1, Nx-1):
        u[n+1, i] = u[n-1, i] + 2*dt * (
            nu * (u[n, i+1] - 2*u[n, i] + u[n, i-1])/(dx*dx)
            - np.pi**2 * nu * np.exp(-t[n]) * np.sin(np.pi*x[i])
            + np.exp(-t[n]) * np.sin(np.pi*x[i])
        )
    
    # Boundary conditions
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[int(Nt/4), :], 'r--', label=f't = {T/4:.1f}')
plt.plot(x, u[int(Nt/2), :], 'g-.', label=f't = {T/2:.1f}')
plt.plot(x, u[-1, :], 'k:', label=f't = {T:.1f}')

plt.title('1D Diffusion Equation - Leapfrog Method')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True)
plt.show()

# Print maximum value for stability check
print(f"Maximum absolute value in solution: {np.max(np.abs(u))}")