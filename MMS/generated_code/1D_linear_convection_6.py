import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 80  # number of spatial points
nt = 200  # number of time steps
dx = 2.0 / (nx - 1)  # spatial step size
dt = 2.0 / (nt - 1)  # time step size
c = 1.0  # wave speed
r = c * dt / dx  # CFL number

# Grid points
x = np.linspace(0, 2, nx)
t = np.linspace(0, 2, nt)

# Initialize solution array
u = np.zeros((nt, nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Source term function
def source(x, t):
    return -np.pi * c * np.exp(-t) * np.cos(np.pi * x) + np.exp(-t) * np.sin(np.pi * x)

# Beam-Warming scheme
def beam_warming_step(u_prev, dt, dx, c, t_current):
    u_new = u_prev.copy()
    
    # Interior points
    for i in range(2, nx):
        u_new[i] = u_prev[i] - c * dt/(2*dx) * (3*u_prev[i] - 4*u_prev[i-1] + u_prev[i-2]) \
                   + dt * source(x[i], t_current)
    
    return u_new

# Time stepping
for n in range(nt-1):
    u[n+1, :] = beam_warming_step(u[n, :], dt, dx, c, t[n])
    u[n+1, 0] = 0  # Enforce boundary conditions
    u[n+1, -1] = 0

# Plot results at key time steps
plt.figure(figsize=(10, 6))
key_times = [0, nt//4, nt//2, nt-1]
labels = ['t = 0', 't = T/4', 't = T/2', 't = T']
colors = ['b', 'g', 'r', 'k']

for i, n in enumerate(key_times):
    plt.plot(x, u[n, :], colors[i], label=labels[i])

plt.title('1D Linear Convection - Beam-Warming Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Print stability condition
print(f"CFL number = {r}")
print("For Beam-Warming scheme, CFL should be ≤ 1 for stability")
if r <= 1:
    print("Solution should be stable")
else:
    print("Warning: Solution might be unstable")