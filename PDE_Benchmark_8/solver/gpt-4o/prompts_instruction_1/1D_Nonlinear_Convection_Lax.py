import numpy as np

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
T = 500

# Discretize spatial domain
dx = dt / nu
x = np.linspace(0, L, int(np.ceil(L / dx)), endpoint=False)
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Time-stepping loop
for n in range(T):
    u_next = np.zeros_like(u)
    # Apply Lax method
    for j in range(nx):
        u_next[j] = 0.5 * (u[(j+1) % nx] + u[(j-1) % nx]) - (dt / (2 * dx)) * (0.5 * u[(j+1) % nx]**2 - 0.5 * u[(j-1) % nx]**2)
    u = u_next

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_1/u_1D_Nonlinear_Convection_Lax.npy', u)