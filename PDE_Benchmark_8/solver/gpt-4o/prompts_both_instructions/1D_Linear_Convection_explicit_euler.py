import numpy as np

# Parameters
c = 1.0  # Convection speed
epsilon_undamped = 0.0
epsilon_damped = 5e-4
x_start, x_end = -5.0, 5.0
N_x = 101
dx = (x_end - x_start) / (N_x - 1)
x = np.linspace(x_start, x_end, N_x)

# Initial condition
u_initial = np.exp(-x**2)

# Time step based on CFL condition
CFL = 0.5
dt = CFL * dx / c

# Total time and number of time steps
t_final = 2.0
N_t = int(t_final / dt)

# Function to apply periodic boundary conditions
def apply_periodic_boundary(u):
    u[0] = u[-2]
    u[-1] = u[1]

# Function to solve the PDE
def solve_pde(epsilon):
    u = u_initial.copy()
    for _ in range(N_t):
        u_new = u.copy()
        # Central difference for spatial derivatives
        u_new[1:-1] = (u[1:-1] - c * dt / (2 * dx) * (u[2:] - u[:-2]) +
                       epsilon * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2]))
        # Apply periodic boundary conditions
        apply_periodic_boundary(u_new)
        u = u_new
    return u

# Solve for undamped case
u_final_undamped = solve_pde(epsilon_undamped)

# Solve for damped case
u_final_damped = solve_pde(epsilon_damped)

# Save the final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_final_undamped_1D_Linear_Convection_explicit_euler.npy', u_final_undamped)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_final_damped_1D_Linear_Convection_explicit_euler.npy', u_final_damped)