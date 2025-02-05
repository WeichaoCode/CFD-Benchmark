import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # Number of spatial points
nt = 1000  # Number of time steps
dx = 2.0 / (nx - 1)  # Spatial step size
dt = 2.0 / (nt - 1)  # Time step size
x = np.linspace(0, 2, nx)  # Spatial domain
t = np.linspace(0, 2, nt)  # Time domain

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

# Beam-Warming scheme
def beam_warming_step(u_prev, dx, dt, t_current):
    u_new = u_prev.copy()
    
    for i in range(2, nx):
        # Beam-Warming discretization for nonlinear convection term
        if u_prev[i] >= 0:
            conv_term = u_prev[i] * (3*u_prev[i] - 4*u_prev[i-1] + u_prev[i-2])/(2*dx)
        else:
            conv_term = u_prev[i] * (-u_prev[i+1] + 4*u_prev[i] - 3*u_prev[i-1])/(2*dx)
            
        # Update solution
        u_new[i] = u_prev[i] - dt * conv_term + dt * source_term(x[i], t_current)
    
    return u_new

# Time integration
for n in range(nt-1):
    u[n+1, :] = beam_warming_step(u[n, :], dx, dt, t[n])
    
    # Apply boundary conditions
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results at key time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4), :], 'r--', label=f't = {t[int(nt/4)]:.2f}')
plt.plot(x, u[int(nt/2), :], 'g-.', label=f't = {t[int(nt/2)]:.2f}')
plt.plot(x, u[-1, :], 'k:', label=f't = {t[-1]:.2f}')

plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Nonlinear Convection - Beam-Warming Scheme')
plt.legend()
plt.grid(True)
plt.show()

# Print maximum values at key time steps for stability check
print(f"Max value at t = 0: {np.max(np.abs(u[0, :])):.4f}")
print(f"Max value at t = T/4: {np.max(np.abs(u[int(nt/4), :])):.4f}")
print(f"Max value at t = T/2: {np.max(np.abs(u[int(nt/2), :])):.4f}")
print(f"Max value at t = T: {np.max(np.abs(u[-1, :])):.4f}")