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

# Time stepping parameters
CFL = 0.5  # CFL number for stability
dt = CFL * dx / c
t_final = 2.0  # Final time
n_steps = int(t_final / dt)

# Central difference for spatial derivatives
def central_diff(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

def laplacian(u, dx):
    return (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx**2)

# Function to compute the right-hand side of the PDE
def rhs(u, epsilon, dx):
    return -c * central_diff(u, dx) + epsilon * laplacian(u, dx)

# Runge-Kutta 4th order time integration
def rk4_step(u, dt, epsilon, dx):
    k1 = rhs(u, epsilon, dx)
    k2 = rhs(u + 0.5 * dt * k1, epsilon, dx)
    k3 = rhs(u + 0.5 * dt * k2, epsilon, dx)
    k4 = rhs(u + dt * k3, epsilon, dx)
    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Function to solve the PDE
def solve_pde(u_initial, epsilon, dx, dt, n_steps):
    u = u_initial.copy()
    for _ in range(n_steps):
        u = rk4_step(u, dt, epsilon, dx)
    return u

# Solve for both undamped and damped cases
u_final_undamped = solve_pde(u_initial, epsilon_undamped, dx, dt, n_steps)
u_final_damped = solve_pde(u_initial, epsilon_damped, dx, dt, n_steps)

# Save the final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_final_undamped_1D_Linear_Convection_rk.npy', u_final_undamped)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_final_damped_1D_Linear_Convection_rk.npy', u_final_damped)