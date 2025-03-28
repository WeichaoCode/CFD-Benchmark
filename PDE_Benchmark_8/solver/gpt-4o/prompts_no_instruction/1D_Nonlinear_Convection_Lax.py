import numpy as np
import math

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
T = 500

# Calculate dx based on the given formula
dx = dt / nu

# Discretize the spatial domain
x = np.linspace(0, L, math.ceil(L / dx))
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Lax method for time integration
u_new = np.zeros_like(u)

for n in range(T):
    # Apply periodic boundary conditions
    u_new[0] = 0.5 * (u[1] + u[-1]) - dt / (2 * dx) * (u[1]**2 / 2 - u[-1]**2 / 2)
    u_new[-1] = 0.5 * (u[0] + u[-2]) - dt / (2 * dx) * (u[0]**2 / 2 - u[-2]**2 / 2)
    
    # Update the solution for the interior points
    for i in range(1, nx - 1):
        u_new[i] = 0.5 * (u[i+1] + u[i-1]) - dt / (2 * dx) * (u[i+1]**2 / 2 - u[i-1]**2 / 2)
    
    # Update the solution
    u[:] = u_new[:]

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_1D_Nonlinear_Convection_Lax.npy', u)