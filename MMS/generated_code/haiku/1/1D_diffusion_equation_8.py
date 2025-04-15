import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
nu = 0.3  # viscosity
x_start, x_end = 0, 2  # spatial domain
t_start, t_end = 0, 2  # temporal domain

# Numerical parameters
nx = 100  # spatial points
nt = 200  # temporal points
dx = (x_end - x_start) / (nx - 1)
dt = (t_end - t_start) / (nt - 1)

# Stability check (von Neumann analysis)
stability_condition = nu * dt / (dx**2)
print(f"Stability condition: {stability_condition}")
if stability_condition > 0.5:
    raise ValueError("Method is unstable. Reduce dt or increase dx.")

# Grid initialization
x = np.linspace(x_start, x_end, nx)
t = np.linspace(t_start, t_end, nt)
u = np.zeros((nt, nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Lax-Friedrichs method
for n in range(nt - 1):
    for i in range(1, nx - 1):
        # Source term
        source = np.pi * nu * np.exp(-t[n]) * np.sin(np.pi * x[i]) - np.exp(-t[n]) * np.sin(np.pi * x[i])
        
        # Lax-Friedrichs scheme
        u[n+1, i] = 0.5 * (u[n, i+1] + u[n, i-1]) - \
                    0.5 * (nu * dt / (dx**2)) * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) + \
                    dt * source

# Plotting
plt.figure(figsize=(10, 6))
plt.title('1D Diffusion Equation: Lax-Friedrichs Method')
plt.xlabel('x')
plt.ylabel('u')

time_steps = [0, nt//4, nt//2, -1]
labels = ['t=0', 't=T/4', 't=T/2', 't=T']

for idx, ts in enumerate(time_steps):
    plt.plot(x, u[ts, :], label=labels[idx])

plt.legend()
plt.grid(True)
plt.show()