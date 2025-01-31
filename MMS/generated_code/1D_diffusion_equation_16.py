import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of spatial domain
T = 2.0  # Total time
nu = 0.3  # Viscosity
Nx = 100  # Number of spatial points
Nt = 1000  # Number of time points
dx = L / (Nx - 1)
dt = T / (Nt - 1)

# Stability check (von Neumann analysis)
stability_parameter = nu * dt / (dx * dx)
print(f"Stability parameter (should be ≤ 0.5): {stability_parameter}")

if stability_parameter > 0.5:
    print("Warning: Solution might be unstable!")

# Initialize grid
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)
u = np.zeros((Nt, Nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Source term
def source_term(x, t):
    return -np.pi**2 * nu * np.exp(-t) * np.sin(np.pi * x) + np.exp(-t) * np.sin(np.pi * x)

# Lax-Friedrichs scheme
def lax_friedrichs_step(u_prev, dx, dt, nu, t_current):
    u_next = np.zeros_like(u_prev)
    
    # Apply scheme for interior points
    for i in range(1, len(u_prev)-1):
        # Diffusion term using Lax-Friedrichs
        diffusion = (nu * dt / (dx**2)) * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1])
        
        # Lax-Friedrichs averaging
        u_next[i] = 0.5 * (u_prev[i+1] + u_prev[i-1]) + diffusion + dt * source_term(x[i], t_current)
    
    return u_next

# Time stepping
for n in range(0, Nt-1):
    u[n+1, :] = lax_friedrichs_step(u[n, :], dx, dt, nu, t[n])
    u[n+1, 0] = 0  # Enforce boundary conditions
    u[n+1, -1] = 0

# Plot results at specified time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[Nt//4, :], 'r--', label=f't = {T/4:.1f}')
plt.plot(x, u[Nt//2, :], 'g-.', label=f't = {T/2:.1f}')
plt.plot(x, u[-1, :], 'k:', label=f't = {T:.1f}')

plt.title('1D Diffusion Equation - Lax-Friedrichs Method')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.grid(True)
plt.legend()
plt.show()

# Print maximum and minimum values
print(f"Maximum value: {np.max(u)}")
print(f"Minimum value: {np.min(u)}")