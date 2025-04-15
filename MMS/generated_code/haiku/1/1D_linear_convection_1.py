import numpy as np
import matplotlib.pyplot as plt

def initial_condition(x):
    return np.sin(np.pi * x)

def boundary_condition(t):
    return 0

def source_term(x, t, c):
    return -np.pi * c * np.exp(-t) * np.cos(np.pi * x) + np.exp(-t) * np.sin(np.pi * x)

def first_order_upwind(nx, nt, dx, dt, c):
    # Initialize solution array
    x = np.linspace(0, 2, nx)
    t = np.linspace(0, 2, nt)
    u = np.zeros((nt, nx))

    # Initial condition
    u[0, :] = initial_condition(x)

    # Stability check (von Neumann analysis)
    cfl = c * dt / dx
    if cfl > 1:
        raise ValueError("Unstable scheme: CFL condition violated")

    # Time stepping
    for n in range(1, nt):
        for i in range(1, nx-1):
            # Upwind scheme
            u[n, i] = u[n-1, i] - c * dt/dx * (u[n-1, i] - u[n-1, i-1]) \
                      + dt * source_term(x[i], t[n-1], c)

        # Boundary conditions
        u[n, 0] = boundary_condition(t[n])
        u[n, -1] = boundary_condition(t[n])

    return x, t, u

# Simulation parameters
nx = 100  # spatial points
nt = 100  # temporal points
c = 1.0   # wave speed
dx = 2 / (nx - 1)
dt = 2 / (nt - 1)

# Solve PDE
x, t, u = first_order_upwind(nx, nt, dx, dt, c)

# Plot results
plt.figure(figsize=(10, 6))
plt.title('1D Linear Convection - First Order Upwind')
plt.xlabel('x')
plt.ylabel('u')

# Select time indices
time_indices = [0, nt//4, nt//2, -1]
time_labels = ['t=0', 't=T/4', 't=T/2', 't=T']

for idx, time_idx in enumerate(time_indices):
    plt.plot(x, u[time_idx, :], label=time_labels[idx])

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

