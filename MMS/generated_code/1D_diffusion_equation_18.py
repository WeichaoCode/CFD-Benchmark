import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.3  # viscosity
L = 2.0   # spatial domain length
T = 2.0   # temporal domain length
Nx = 100  # number of spatial points
dx = L / (Nx - 1)
dt = 0.001  # time step (chosen based on stability analysis)
Nt = int(T/dt)  # number of time steps

# Spatial grid
x = np.linspace(0, L, Nx)

# Initialize solution array
u = np.zeros((Nt+1, Nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Beam-Warming scheme coefficient
r = nu * dt / (dx * dx)

# Stability check
if r > 0.5:
    print("Warning: Scheme might be unstable! Reduce dt or increase dx")

# Time integration
for n in range(0, Nt):
    for i in range(1, Nx-1):
        # Source term
        source = -np.pi**2 * nu * np.exp(-n*dt) * np.sin(np.pi * x[i]) + \
                 np.exp(-n*dt) * np.sin(np.pi * x[i])
        
        # Beam-Warming scheme
        u[n+1, i] = u[n, i] + \
                    r * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) + \
                    dt * source

    # Boundary conditions
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results at specified time steps
plt.figure(figsize=(10, 6))
t_plot = [0, int(Nt/4), int(Nt/2), Nt]
labels = ['t = 0', 't = T/4', 't = T/2', 't = T']
colors = ['b', 'g', 'r', 'k']

for idx, t in enumerate(t_plot):
    plt.plot(x, u[t, :], colors[idx], label=labels[idx])

plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('1D Diffusion Equation - Beam-Warming Scheme')
plt.grid(True)
plt.legend()
plt.show()

# Calculate and print maximum value for stability monitoring
print(f"Maximum solution value: {np.max(np.abs(u))}")

# Calculate L2 error norm at final time
exact_final = np.exp(-T) * np.sin(np.pi * x)
error = np.sqrt(np.mean((u[-1, :] - exact_final)**2))
print(f"L2 error norm at final time: {error}")