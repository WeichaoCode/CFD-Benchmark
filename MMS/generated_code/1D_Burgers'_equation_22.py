import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0 / (nx - 1)  # spatial step size
dt = 2.0 / (nt - 1)  # time step size
nu = 0.07  # viscosity

# Initialize arrays
x = np.linspace(0, 2, nx)
t = np.linspace(0, 2, nt)
u = np.zeros((nt, nx))

# Source term function
def source(x, t):
    return (-np.pi**2 * nu * np.exp(-t) * np.sin(np.pi*x) + 
            np.exp(-t) * np.sin(np.pi*x) - 
            np.pi * np.exp(-2*t) * np.sin(np.pi*x) * np.cos(np.pi*x))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Check stability (CFL condition)
c = dt/(dx**2)
if c > 0.5:
    print("Warning: Solution might be unstable!")
    print(f"CFL number: {c}")

# Lax-Friedrichs scheme
for n in range(0, nt-1):
    for i in range(1, nx-1):
        # Convective term
        conv = u[n,i] * (u[n,i+1] - u[n,i-1])/(2*dx)
        
        # Diffusive term
        diff = nu * (u[n,i+1] - 2*u[n,i] + u[n,i-1])/(dx**2)
        
        # Source term
        s = source(x[i], t[n])
        
        # Lax-Friedrichs scheme
        u[n+1,i] = 0.5*(u[n,i+1] + u[n,i-1]) - \
                   dt/(2*dx) * (0.5*u[n,i+1]**2 - 0.5*u[n,i-1]**2) + \
                   nu*dt/(dx**2) * (u[n,i+1] - 2*u[n,i] + u[n,i-1]) + \
                   dt*s

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, u[0,:], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4),:], 'r--', label=f't = {t[int(nt/4)]:.2f}')
plt.plot(x, u[int(nt/2),:], 'g-.', label=f't = {t[int(nt/2)]:.2f}')
plt.plot(x, u[-1,:], 'k:', label=f't = {t[-1]:.2f}')

plt.xlabel('x')
plt.ylabel('u')
plt.title("1D Burgers' Equation - Lax-Friedrichs Method")
plt.legend()
plt.grid(True)
plt.show()

# Print maximum value for stability check
print(f"Maximum value in solution: {np.max(np.abs(u))}")