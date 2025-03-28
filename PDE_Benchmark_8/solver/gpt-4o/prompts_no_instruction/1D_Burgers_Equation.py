import numpy as np

# Parameters
nu = 0.07
nx = 101
nt = 100
x_start = 0.0
x_end = 2.0 * np.pi
dx = (x_end - x_start) / (nx - 1)
dt = dx * nu

# Discretize the spatial domain
x = np.linspace(x_start, x_end, nx)

# Initial condition
phi = np.exp(-x**2 / (4 * nu)) + np.exp(-(x - 2 * np.pi)**2 / (4 * nu))
u = -2 * nu / phi * np.gradient(phi, dx) + 4

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    # Apply periodic boundary conditions
    u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2])
    u[-1] = u[0]
    # Update the interior points
    for i in range(1, nx - 1):
        u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i - 1]) + nu * dt / dx**2 * (un[i + 1] - 2 * un[i] + un[i - 1])

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_1D_Burgers_Equation.npy', u)