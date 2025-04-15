import numpy as np
import matplotlib.pyplot as plt

def initial_condition(x):
    return np.sin(np.pi * x)

def source_term(x, t):
    return -np.pi * np.exp(-t) * np.cos(np.pi * x) + np.exp(-t) * np.sin(np.pi * x)

def lax_friedrichs_1d_convection(c=1.0, T=2.0, x_start=0.0, x_end=2.0, nx=100, nt=200):
    # Grid setup
    dx = (x_end - x_start) / (nx - 1)
    dt = T / (nt - 1)
    
    # CFL condition check
    cfl = c * dt / dx
    print(f"CFL number: {cfl}")
    
    # Grid initialization
    x = np.linspace(x_start, x_end, nx)
    t = np.linspace(0, T, nt)
    u = np.zeros((nt, nx))
    
    # Initial condition
    u[0, :] = initial_condition(x)
    
    # Boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0
    
    # Lax-Friedrichs scheme
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            source = source_term(x[i], t[n])
            u[n+1, i] = 0.5 * (u[n, i+1] + u[n, i-1]) - \
                        0.5 * c * dt / dx * (u[n, i+1] - u[n, i-1]) + \
                        dt * source
    
    return x, t, u

# Solve the PDE
x, t, u = lax_friedrichs_1d_convection()

# Plot results at key time steps
plt.figure(figsize=(10, 6))
plt.title('1D Linear Convection - Lax-Friedrichs Method')
plt.xlabel('x')
plt.ylabel('u')

time_indices = [0, int(len(t)/4), int(len(t)/2), -1]
time_labels = ['t=0', 't=T/4', 't=T/2', 't=T']

for idx, time_idx in enumerate(time_indices):
    plt.plot(x, u[time_idx, :], label=time_labels[idx])

plt.legend()
plt.grid(True)
plt.show()

