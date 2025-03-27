import numpy as np

# Parameters
nx = 101
nt = 100
nu = 0.07
x_start = 0
x_end = 2 * np.pi
dx = (x_end - x_start) / (nx - 1)
dt = dx * nu

# Spatial grid
x = np.linspace(x_start, x_end, nx)

# Initial condition
phi = np.exp(-x**2 / (4 * nu)) + np.exp(-(x - 2 * np.pi)**2 / (4 * nu))
u = -2 * nu / phi * np.gradient(phi, dx) + 4

# Time-stepping loop
for n in range(nt):
    u_old = u.copy()
    # Upwind scheme for convection term
    for i in range(1, nx-1):
        u[i] = u_old[i] - dt / dx * u_old[i] * (u_old[i] - u_old[i-1]) + \
               nu * dt / dx**2 * (u_old[i+1] - 2 * u_old[i] + u_old[i-1])
    # Periodic boundary conditions
    u[0] = u[-1]
    u[-1] = u[0]

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_1/u_1D_Burgers_Equation.npy', u)