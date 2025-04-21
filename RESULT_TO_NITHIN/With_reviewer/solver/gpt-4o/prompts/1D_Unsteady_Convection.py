import numpy as np

# Parameters
u = 0.2  # m/s
x_start, x_end = 0.0, 2.0
t_start, t_end = 0.0, 2.5
nx = 100  # number of spatial points
nt = 250  # number of time steps
dx = (x_end - x_start) / (nx - 1)
dt = (t_end - t_start) / nt

# Stability condition
assert u * dt / dx <= 1, "CFL condition not satisfied!"

# Spatial grid
x = np.linspace(x_start, x_end, nx)

# Initial condition
m = 0.5
s = 0.1
phi = np.exp(-((x - m) / s) ** 2)

# Time-stepping loop
for n in range(nt):
    phi_new = phi.copy()
    for i in range(1, nx - 1):
        phi_new[i] = phi[i] - u * dt / dx * (phi[i] - phi[i - 1])
    # Apply boundary conditions
    phi_new[0] = 0
    phi_new[-1] = 0
    phi = phi_new

# Save the final solution
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/phi_1D_Unsteady_Convection.npy', phi)