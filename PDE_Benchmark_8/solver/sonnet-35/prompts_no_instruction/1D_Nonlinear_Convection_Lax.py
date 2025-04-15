import numpy as np
import math

# Problem parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
x = np.linspace(0, L, math.ceil(L / dx))
T = 500

# Initial condition
def initial_condition(x):
    return np.sin(x) + 0.5 * np.sin(0.5 * x)

# Allocate solution array
u = initial_condition(x)

# Lax method for time integration
for _ in range(T):
    u_old = u.copy()
    
    # Periodic boundary conditions
    u[0] = u_old[-1]
    u[-1] = u_old[0]
    
    # Lax method update
    u[1:-1] = 0.5 * (u_old[2:] + u_old[:-2]) - 0.5 * dt/dx * u_old[1:-1] * (u_old[2:] - u_old[:-2])

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_1D_Nonlinear_Convection_Lax.npy', u)