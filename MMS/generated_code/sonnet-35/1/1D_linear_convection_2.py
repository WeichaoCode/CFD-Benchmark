import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0 / (nx - 1)  # spatial step size
dt = 2.0 / (nt - 1)  # time step size
c = 1.0  # wave speed

# Check CFL condition for stability
CFL = c * dt / dx
print(f"CFL number: {CFL}")
if CFL > 1:
    print("Warning: Solution might be unstable!")

# Initialize arrays
x = np.linspace(0, 2, nx)
t = np.linspace(0, 2, nt)
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Set boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Source term function
def source(x, t):
    return -np.pi * c * np.exp(-t) * np.cos(np.pi * x) + np.exp(-t) * np.sin(np.pi * x)

# Lax-Friedrichs scheme
for n in range(0, nt-1):
    for i in range(1, nx-1):
        u[n+1, i] = 0.5 * (u[n, i+1] + u[n, i-1]) - \
                    0.5 * c * dt/dx * (u[n, i+1] - u[n, i-1]) + \
                    dt * source(x[i], t[n])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4), :], 'r--', label=f't = {t[int(nt/4)]:.2f}')
plt.plot(x, u[int(nt/2), :], 'g-.', label=f't = {t[int(nt/2)]:.2f}')
plt.plot(x, u[-1, :], 'k:', label=f't = {t[-1]:.2f}')

plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Linear Convection - Lax-Friedrichs Scheme')
plt.legend()
plt.grid(True)
plt.show()

# Print maximum absolute value to check stability
print(f"Maximum absolute value: {np.max(np.abs(u))}")

