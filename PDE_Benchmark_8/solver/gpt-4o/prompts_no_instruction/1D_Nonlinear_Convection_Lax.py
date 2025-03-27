import numpy as np

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
T = 500  # number of time steps

# Discretize the spatial domain
x = np.linspace(0, L, int(np.ceil(L / dx)), endpoint=False)
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Lax method for time integration
for n in range(T):
    u_next = np.zeros_like(u)
    for i in range(nx):
        u_next[i] = 0.5 * (u[i-1] + u[(i+1) % nx]) - (dt / (2 * dx)) * (u[(i+1) % nx]**2 / 2 - u[i-1]**2 / 2)
    u = u_next

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_1D_Nonlinear_Convection_Lax.npy', u)