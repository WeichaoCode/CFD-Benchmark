import numpy as np

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
T = 500  # Number of time steps

# Discretize the spatial domain
x = np.linspace(0, L, int(np.ceil(L / dx)), endpoint=False)
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# MacCormack method
for n in range(T):
    # Predictor step
    u_star = np.empty_like(u)
    for j in range(nx):
        jp1 = (j + 1) % nx  # Periodic boundary condition
        u_star[j] = u[j] - dt / dx * (0.5 * u[jp1]**2 - 0.5 * u[j]**2)
    
    # Corrector step
    u_new = np.empty_like(u)
    for j in range(nx):
        jm1 = (j - 1) % nx  # Periodic boundary condition
        u_new[j] = 0.5 * (u[j] + u_star[j] - dt / dx * (0.5 * u_star[j]**2 - 0.5 * u_star[jm1]**2))
    
    # Update solution
    u = u_new

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_1/u_1D_Nonlinear_Convection_MK.npy', u)