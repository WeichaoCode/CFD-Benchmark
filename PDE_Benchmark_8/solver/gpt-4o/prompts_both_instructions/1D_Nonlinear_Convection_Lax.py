import numpy as np
import math

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
T = 500

# Calculate dx based on dt and nu
dx = dt / nu

# Discretize the spatial domain
x = np.linspace(0, L, math.ceil(L / dx), endpoint=False)
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Lax method time-stepping
for n in range(T):
    u_next = np.zeros_like(u)
    # Apply Lax method
    for j in range(nx):
        u_next[j] = 0.5 * (u[(j + 1) % nx] + u[(j - 1) % nx]) - \
                    (dt / (2 * dx)) * ((u[(j + 1) % nx]**2 / 2) - (u[(j - 1) % nx]**2 / 2))
    u = u_next

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_1D_Nonlinear_Convection_Lax.npy', u)