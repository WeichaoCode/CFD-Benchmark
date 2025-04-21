import numpy as np

# Parameters
nx = 100
nt = 500
dx = 2 * np.pi / nx
dt = 5 / nt
x = np.linspace(0, 2 * np.pi, nx, endpoint=False)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Numerical solution using Lax-Friedrichs scheme
for n in range(nt):
    u_old = u.copy()
    u[1:-1] = 0.5 * (u_old[2:] + u_old[:-2]) - 0.25 * dt/dx * (u_old[2:]**2 - u_old[:-2]**2)
    # Periodic boundary conditions
    u[0] = 0.5 * (u_old[1] + u_old[-1]) - 0.25 * dt/dx * (u_old[1]**2 - u_old[-1]**2)
    u[-1] = 0.5 * (u_old[0] + u_old[-2]) - 0.25 * dt/dx * (u_old[0]**2 - u_old[-2]**2)

# Save the solution at the final time step
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_1D_Nonlinear_Convection.npy', u)