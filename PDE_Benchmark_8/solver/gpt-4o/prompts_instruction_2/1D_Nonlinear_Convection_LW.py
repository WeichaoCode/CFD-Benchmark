import numpy as np
import math

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
T = 500

# Discretize the spatial domain
x = np.linspace(0, L, math.ceil(L / dx))
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Lax-Wendroff method
def lax_wendroff(u, dt, dx, nx):
    u_new = np.zeros_like(u)
    for i in range(1, nx - 1):
        u_new[i] = (u[i] - 0.5 * dt / dx * u[i] * (u[i+1] - u[i-1]) +
                    0.5 * (dt / dx)**2 * u[i]**2 * (u[i+1] - 2*u[i] + u[i-1]))
    # Apply periodic boundary conditions
    u_new[0] = (u[0] - 0.5 * dt / dx * u[0] * (u[1] - u[-2]) +
                0.5 * (dt / dx)**2 * u[0]**2 * (u[1] - 2*u[0] + u[-2]))
    u_new[-1] = u_new[0]
    return u_new

# Time integration
for _ in range(T):
    u = lax_wendroff(u, dt, dx, nx)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/u_1D_Nonlinear_Convection_LW.npy', u)