import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
c = 1.0  # Wave speed

# Grid parameters
nx = 100  # Number of spatial points
nt = 200  # Number of time points
dx = L/(nx-1)  # Spatial step
dt = T/(nt-1)  # Time step

# Check CFL condition for stability
CFL = c*dt/dx
print(f"CFL number: {CFL}")
if CFL > 1:
    print("Warning: Solution might be unstable!")

# Initialize arrays
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(np.pi*x)

# Set boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Source term function
def source(x, t):
    return -np.pi*c*np.exp(-t)*np.cos(np.pi*x) + np.exp(-t)*np.sin(np.pi*x)

# Lax-Friedrichs scheme
for n in range(0, nt-1):
    for i in range(1, nx-1):
        u[n+1, i] = 0.5*(u[n, i+1] + u[n, i-1]) - \
                    (c*dt/(2*dx))*(u[n, i+1] - u[n, i-1]) + \
                    dt*source(x[i], t[n])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[nt//4, :], 'g-', label=f't = {T/4:.1f}')
plt.plot(x, u[nt//2, :], 'r-', label=f't = {T/2:.1f}')
plt.plot(x, u[-1, :], 'k-', label=f't = {T:.1f}')

plt.title('1D Linear Convection - Lax-Friedrichs Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Print maximum and minimum values for stability check
print(f"Maximum value: {np.max(u)}")
print(f"Minimum value: {np.min(u)}")