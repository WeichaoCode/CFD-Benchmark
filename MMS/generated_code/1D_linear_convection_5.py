import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
c = 1.0  # Wave speed
nx = 100  # Number of spatial points
nt = 200  # Number of time points

# Grid parameters
dx = L / (nx-1)
dt = T / (nt-1)
CFL = c * dt / dx

# Check stability (CFL condition)
if CFL > 1:
    print("Warning: Solution might be unstable! CFL =", CFL)

# Create grid
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Initialize solution array
u = np.zeros((nt, nx))

# Source term function
def source(x, t):
    return -np.pi * c * np.exp(-t) * np.cos(np.pi * x) + np.exp(-t) * np.sin(np.pi * x)

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Set boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Lax-Wendroff scheme
for n in range(0, nt-1):
    for i in range(1, nx-1):
        # Lax-Wendroff formula
        u[n+1, i] = u[n, i] - 0.5 * CFL * (u[n, i+1] - u[n, i-1]) + \
                    0.5 * CFL**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) + \
                    dt * source(x[i], t[n])

# Plot results at specific time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4), :], 'r--', label=f't = {T/4:.2f}')
plt.plot(x, u[int(nt/2), :], 'g-.', label=f't = {T/2:.2f}')
plt.plot(x, u[-1, :], 'k:', label=f't = {T:.2f}')

plt.title('1D Linear Convection - Lax-Wendroff Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Print maximum absolute value to check stability
print("Maximum absolute value:", np.max(np.abs(u)))