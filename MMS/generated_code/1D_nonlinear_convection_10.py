import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0 / (nx - 1)  # spatial step size
dt = 2.0 / (nt - 1)  # time step size
x = np.linspace(0, 2, nx)  # spatial domain
t = np.linspace(0, 2, nt)  # temporal domain

# Check CFL condition for stability
c = dt / dx
print(f"CFL number: {c}")

# Initialize solution array
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Set boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Source term function
def source_term(x, t):
    return np.exp(-t) * np.sin(np.pi * x) - np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x)

# Lax-Friedrichs scheme
def lax_friedrichs_step(u_prev, dt, dx, t):
    u_next = np.zeros_like(u_prev)
    
    # Apply scheme for interior points
    for i in range(1, len(u_prev)-1):
        # Lax-Friedrichs discretization
        u_next[i] = 0.5 * (u_prev[i+1] + u_prev[i-1]) - \
                    0.5 * dt/dx * (u_prev[i+1]**2/2 - u_prev[i-1]**2/2) + \
                    dt * source_term(x[i], t)
    
    # Apply boundary conditions
    u_next[0] = 0
    u_next[-1] = 0
    
    return u_next

# Time integration
for n in range(0, nt-1):
    u[n+1] = lax_friedrichs_step(u[n], dt, dx, t[n])

# Plot results at key time steps
plt.figure(figsize=(10, 6))
key_times = [0, int(nt/4), int(nt/2), nt-1]
labels = ['t = 0', 't = T/4', 't = T/2', 't = T']
colors = ['b', 'r', 'g', 'k']

for i, n in enumerate(key_times):
    plt.plot(x, u[n], colors[i], label=labels[i])

plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Nonlinear Convection - Lax-Friedrichs Scheme')
plt.legend()
plt.grid(True)
plt.show()

# Print maximum and minimum values for stability check
print(f"Maximum value: {np.max(u)}")
print(f"Minimum value: {np.min(u)}")