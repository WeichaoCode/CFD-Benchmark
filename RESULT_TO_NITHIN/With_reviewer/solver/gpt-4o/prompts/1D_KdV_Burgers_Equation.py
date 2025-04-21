import numpy as np

# Parameters
a = 1e-4
b = 2e-4
L = 10.0
T = 10.0
n = 20

# Discretization
Nx = 200  # Number of spatial points
Nt = 1000  # Number of time steps
dx = L / Nx
dt = T / Nt

# Spatial and temporal grids
x = np.linspace(0, L, Nx, endpoint=False)
t = np.linspace(0, T, Nt)

# Initial condition
u = 0.5 / n * np.log(1 + (np.cosh(n) ** 2) / (np.cosh(n * (x - 0.2 * L)) ** 2))

# Time-stepping loop using finite difference method
for _ in range(Nt):
    # Compute spatial derivatives
    u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx ** 2)
    u_xxx = (np.roll(u_x, -1) - 2 * u_x + np.roll(u_x, 1)) / (dx ** 2)

    # Update u using a forward time, centered space (FTCS) scheme
    u_new = u - dt * (0.5 * u * u_x) + dt * a * u_xx + dt * b * u_xxx

    # Check for NaNs or Infs and break if found
    if np.any(np.isnan(u_new)) or np.any(np.isinf(u_new)):
        print("Numerical instability detected. Stopping simulation.")
        break

    u = u_new

# Save the final solution
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_1D_KdV_Burgers_Equation.npy', u)