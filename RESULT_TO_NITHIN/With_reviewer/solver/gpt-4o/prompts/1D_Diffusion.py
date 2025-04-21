import numpy as np

# Parameters
nu = 0.3  # diffusion coefficient
x_start, x_end = 0.0, 2.0  # spatial domain
t_start, t_end = 0.0, 0.0333  # temporal domain
nx = 101  # number of spatial points
nt = 1000  # number of time steps
dx = (x_end - x_start) / (nx - 1)  # spatial step size
dt = (t_end - t_start) / nt  # time step size

# Stability condition
assert dt <= dx**2 / (2 * nu), "Stability condition violated!"

# Discretized spatial domain
x = np.linspace(x_start, x_end, nx)

# Initial condition
u = np.ones(nx)
u[int(0.5 / dx):int(1.0 / dx) + 1] = 2

# Time-stepping loop
for n in range(nt):
    u_new = u.copy()
    for i in range(1, nx - 1):
        u_new[i] = u[i] + nu * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    u = u_new

# Save the final solution
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_1D_Diffusion.npy', u)