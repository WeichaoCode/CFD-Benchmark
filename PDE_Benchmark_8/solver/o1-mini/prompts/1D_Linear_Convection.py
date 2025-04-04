import numpy as np

# Parameters
c = 1.0          # Convection speed
epsilon = 5e-4   # Damping factor
x_start = -5.0
x_end = 5.0
t_start = 0.0
t_end = 10.0
Nx = 1000        # Number of spatial points
dx = (x_end - x_start) / Nx
x = np.linspace(x_start, x_end, Nx, endpoint=False)
u = np.exp(-x**2)  # Initial condition

# Time step based on stability criteria
dt_adv = dx / c
dt_diff = 0.5 * dx**2 / epsilon if epsilon != 0 else np.inf
dt = 0.4 * min(dt_adv, dt_diff)
Nt = int((t_end - t_start) / dt) + 1

# Time integration
for _ in range(Nt):
    u_next = np.empty_like(u)
    # Periodic boundary conditions
    u_minus = np.roll(u, 1)
    u_plus = np.roll(u, -1)
    # Upwind scheme for convection
    conv = c * (u - u_minus) / dx
    # Central difference for diffusion
    diff = epsilon * (u_plus - 2 * u + u_minus) / dx**2
    # Update solution
    u_next = u - dt * conv + dt * diff
    u = u_next

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/u_1D_Linear_Convection.npy', u)