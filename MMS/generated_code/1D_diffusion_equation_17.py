import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.3  # viscosity
L = 2.0   # length of domain
T = 2.0   # total time
nx = 100  # number of spatial points
nt = 1000 # number of time steps

# Grid
dx = L / (nx-1)
dt = T / (nt-1)
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Stability check (von Neumann analysis)
r = nu * dt / (dx**2)
print(f"Stability parameter r = {r}")
if r > 0.5:
    print("Warning: Solution might be unstable! Reduce dt or increase dx.")

# Initialize solution array
u = np.zeros((nt, nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Time stepping
for n in range(0, nt-1):
    # Interior points
    for i in range(1, nx-1):
        # Lax-Wendroff scheme
        u[n+1, i] = u[n, i] + dt * (
            nu * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2 +
            np.pi**2 * nu * np.exp(-t[n]) * np.sin(np.pi * x[i]) -
            np.exp(-t[n]) * np.sin(np.pi * x[i])
        )
    
    # Boundary conditions
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4), :], 'r--', label=f't = {T/4:.2f}')
plt.plot(x, u[int(nt/2), :], 'g-.', label=f't = {T/2:.2f}')
plt.plot(x, u[-1, :], 'k:', label=f't = {T:.2f}')

plt.title('1D Diffusion Equation - Lax-Wendroff Method')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.grid(True)
plt.legend()
plt.show()

# Calculate and print maximum value for stability check
print(f"Maximum value in solution: {np.max(np.abs(u))}")