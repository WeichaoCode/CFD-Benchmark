import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
nu = 0.07  # Viscosity

# Grid parameters
Nx = 100  # Number of spatial points
Nt = 1000  # Number of time steps
dx = L / (Nx - 1)
dt = T / Nt

# Grid points
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Stability check (von Neumann analysis)
CFL = dt / dx
Re = dx / nu
if CFL > 1 or Re > 2:
    print("Warning: Solution might be unstable!")
    print(f"CFL number: {CFL}")
    print(f"Grid Reynolds number: {Re}")

# Initialize solution array
u = np.zeros((Nt, Nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Source term function
def source(x, t):
    return (-np.pi**2 * nu * np.exp(-t) * np.sin(np.pi * x) + 
            np.exp(-t) * np.sin(np.pi * x) - 
            np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x))

# First time step using Forward Euler (needed for Leapfrog)
for i in range(1, Nx-1):
    u[1, i] = u[0, i] - 0.5 * dt * u[0, i] * (u[0, i+1] - u[0, i-1]) / dx + \
              nu * dt * (u[0, i+1] - 2*u[0, i] + u[0, i-1]) / dx**2 + \
              dt * source(x[i], 0)

# Leapfrog scheme
for n in range(1, Nt-1):
    for i in range(1, Nx-1):
        # Convective term
        conv = -0.25 * dt * u[n, i] * (u[n, i+1] - u[n, i-1]) / dx
        
        # Diffusive term
        diff = nu * dt * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2
        
        # Source term
        src = dt * source(x[i], t[n])
        
        u[n+1, i] = u[n-1, i] + 2 * (conv + diff + src)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[Nt//4, :], 'r--', label=f't = {T/4}')
plt.plot(x, u[Nt//2, :], 'g-.', label=f't = {T/2}')
plt.plot(x, u[-1, :], 'k:', label=f't = {T}')

plt.title("1D Burgers' Equation - Leapfrog Method")
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()