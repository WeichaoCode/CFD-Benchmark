import numpy as np

# Parameters
c = 1.0  # Convection speed
epsilon_undamped = 0.0  # Damping factor for undamped case
epsilon_damped = 5e-4  # Damping factor for damped case
x_start, x_end = -5.0, 5.0  # Spatial domain
N_x = 101  # Number of spatial grid points
dx = (x_end - x_start) / (N_x - 1)  # Spatial step size
x = np.linspace(x_start, x_end, N_x)  # Spatial grid

# Initial condition
u_initial = np.exp(-x**2)

# Time step based on CFL condition
CFL = 0.5  # CFL number
dt = CFL * dx / c  # Time step size

# Number of time steps
T_final = 2.0  # Final time
N_t = int(T_final / dt)  # Number of time steps

# Function to compute spatial derivatives
def compute_derivatives(u, epsilon):
    # Central difference for first derivative
    du_dx = np.zeros_like(u)
    du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    du_dx[0] = (u[1] - u[-1]) / (2 * dx)  # Periodic BC
    du_dx[-1] = (u[0] - u[-2]) / (2 * dx)  # Periodic BC

    # Central difference for second derivative
    d2u_dx2 = np.zeros_like(u)
    d2u_dx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2)
    d2u_dx2[0] = (u[1] - 2 * u[0] + u[-1]) / (dx**2)  # Periodic BC
    d2u_dx2[-1] = (u[0] - 2 * u[-1] + u[-2]) / (dx**2)  # Periodic BC

    return -c * du_dx + epsilon * d2u_dx2

# Runge-Kutta 4th order time integration
def rk4_step(u, dt, epsilon):
    k1 = compute_derivatives(u, epsilon)
    k2 = compute_derivatives(u + 0.5 * dt * k1, epsilon)
    k3 = compute_derivatives(u + 0.5 * dt * k2, epsilon)
    k4 = compute_derivatives(u + dt * k3, epsilon)
    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Function to solve the PDE
def solve_pde(epsilon):
    u = u_initial.copy()
    for _ in range(N_t):
        u = rk4_step(u, dt, epsilon)
    return u

# Solve for both undamped and damped cases
u_final_undamped = solve_pde(epsilon_undamped)
u_final_damped = solve_pde(epsilon_damped)

# Save the final solutions to .npy files
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_final_undamped_1D_Linear_Convection_rk.npy', u_final_undamped)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_final_damped_1D_Linear_Convection_rk.npy', u_final_damped)