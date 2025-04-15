import numpy as np

# Parameters
nu = 0.3
x_start, x_end = 0.0, 2.0
t_start, t_end = 0.0, 0.0333
nx = 101  # number of spatial points
nt = 100  # number of time steps
dx = (x_end - x_start) / (nx - 1)
dt = (t_end - t_start) / nt

# Stability condition
assert nu * dt / dx**2 <= 0.5, "Stability condition not met!"

# Discretized domain
x = np.linspace(x_start, x_end, nx)
u = np.ones(nx)

# Initial condition
u[(x >= 0.5) & (x <= 1.0)] = 2

# Time-stepping loop
for n in range(nt):
    u_new = u.copy()
    for i in range(1, nx-1):
        u_new[i] = u[i] + nu * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    u = u_new

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_1D_Diffusion.npy', u)