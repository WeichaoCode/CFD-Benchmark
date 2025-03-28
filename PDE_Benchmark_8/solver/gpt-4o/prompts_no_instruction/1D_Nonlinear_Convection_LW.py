import numpy as np
import math

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
T = 500

# Calculate dx based on the given formula
dx = dt / nu
nx = math.ceil(L / dx)

# Create the spatial grid
x = np.linspace(0, L, nx, endpoint=False)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Lax-Wendroff method
for n in range(T):
    u_next = np.zeros_like(u)
    for i in range(nx):
        # Periodic boundary conditions
        ip1 = (i + 1) % nx
        im1 = (i - 1) % nx
        
        # Lax-Wendroff scheme
        u_next[i] = (u[i] - 0.5 * dt / dx * u[i] * (u[ip1] - u[im1]) +
                     0.5 * (dt / dx)**2 * u[i]**2 * (u[ip1] - 2 * u[i] + u[im1]))
    
    # Update the solution
    u = u_next

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_1D_Nonlinear_Convection_LW.npy', u)