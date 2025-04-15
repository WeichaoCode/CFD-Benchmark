import numpy as np

# Parameters
nu = 0.07
nx = 101
nt = 100
dx = 2 * np.pi / (nx - 1)
dt = dx * nu

# Spatial domain
x = np.linspace(0, 2 * np.pi, nx)

# Initial condition
phi = np.exp(-x**2 / (4 * nu)) + np.exp(-(x - 2 * np.pi)**2 / (4 * nu))
u = -2 * nu / phi * (-x / (2 * nu) * np.exp(-x**2 / (4 * nu)) - (x - 2 * np.pi) / (2 * nu) * np.exp(-(x - 2 * np.pi)**2 / (4 * nu))) + 4

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    # Central difference for spatial derivative, forward difference for time derivative
    u[1:-1] = (un[1:-1] - dt / (2 * dx) * un[1:-1] * (un[2:] - un[:-2]) +
               nu * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2]))
    # Periodic boundary conditions
    u[0] = (un[0] - dt / (2 * dx) * un[0] * (un[1] - un[-2]) +
            nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2]))
    u[-1] = u[0]

# Save the final solution
save_values = ['u']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_1D_Burgers_Equation.npy', u)