import numpy as np

# Parameters
L = 2 * np.pi  # Length of the domain
T = 5.0        # Total time
nx = 100       # Number of spatial points
nt = 500       # Number of time steps
dx = L / nx    # Spatial step size
dt = T / nt    # Time step size

# Discretized spatial domain
x = np.linspace(0, L, nx, endpoint=False)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Time-stepping loop using Lax-Friedrichs method
for n in range(nt):
    u_next = np.zeros_like(u)
    for i in range(nx):
        u_next[i] = 0.5 * (u[i-1] + u[(i+1) % nx]) - dt / (2 * dx) * (u[(i+1) % nx]**2 / 2 - u[i-1]**2 / 2)
    u = u_next

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_1D_Nonlinear_Convection.npy', u)