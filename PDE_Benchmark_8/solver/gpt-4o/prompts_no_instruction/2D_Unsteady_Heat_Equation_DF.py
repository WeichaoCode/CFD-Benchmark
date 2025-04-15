import numpy as np

# Parameters
alpha = 0.01  # thermal diffusivity
Q0 = 200.0  # source term strength
sigma = 0.1  # source term spread
r = 0.25  # stability parameter for DuFort-Frankel
t_max = 3.0  # maximum time

# Domain
nx, ny = 41, 41
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = r * dx**2 / alpha

# Initial condition
X, Y = np.meshgrid(x, y)
T_initial = 1 + 200 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Boundary condition
T_boundary = 1

# Time stepping
nt = int(t_max / dt) + 1
T_old = T_initial.copy()
T_new = T_initial.copy()

# Source term
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# DuFort-Frankel method
for n in range(1, nt):
    T_new[1:-1, 1:-1] = (
        (1 - 2 * r) * T_old[1:-1, 1:-1]
        + 2 * r * (T_old[2:, 1:-1] + T_old[:-2, 1:-1] + T_old[1:-1, 2:] + T_old[1:-1, :-2])
        - T_new[1:-1, 1:-1]
        + 2 * dt * q[1:-1, 1:-1] / alpha
    ) / (1 + 2 * r)

    # Apply boundary conditions
    T_new[0, :] = T_boundary
    T_new[-1, :] = T_boundary
    T_new[:, 0] = T_boundary
    T_new[:, -1] = T_boundary

    # Update old temperature
    T_old, T_new = T_new, T_old

# Save the final temperature field
save_values = ['T']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/T_old_2D_Unsteady_Heat_Equation_DF.npy', T_old)