import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0/(nx-1)  # spatial step size
dt = 2.0/nt  # time step size
x = np.linspace(0, 2, nx)  # spatial domain
t = np.linspace(0, 2, nt)  # temporal domain

# Initialize solution array
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Set boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Source term function
def source(x, t):
    return np.exp(-t) * np.sin(np.pi * x) - np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x)

# First Order Upwind scheme
def upwind(u_prev, dt, dx, t):
    u_new = np.copy(u_prev)
    for i in range(1, nx-1):
        if u_prev[i] >= 0:
            u_new[i] = u_prev[i] - dt/dx * u_prev[i] * (u_prev[i] - u_prev[i-1]) + dt * source(x[i], t)
        else:
            u_new[i] = u_prev[i] - dt/dx * u_prev[i] * (u_prev[i+1] - u_prev[i]) + dt * source(x[i], t)
    return u_new

# Time stepping
for n in range(0, nt-1):
    u[n+1, :] = upwind(u[n, :], dt, dx, t[n])
    u[n+1, 0] = 0  # Enforce boundary conditions
    u[n+1, -1] = 0

# Plot results at specified time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4), :], 'r--', label=f't = {t[int(nt/4)]:.2f}')
plt.plot(x, u[int(nt/2), :], 'g-.', label=f't = {t[int(nt/2)]:.2f}')
plt.plot(x, u[-1, :], 'k:', label=f't = {t[-1]:.2f}')

plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Nonlinear Convection - First Order Upwind')
plt.legend()
plt.grid(True)
plt.show()

# Print max CFL number for stability check
max_velocity = np.max(np.abs(u))
cfl = max_velocity * dt/dx
print(f"Maximum CFL number: {cfl:.3f}")

