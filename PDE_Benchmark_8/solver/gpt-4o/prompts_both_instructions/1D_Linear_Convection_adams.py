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
t_final = 2.0  # Final time
N_t = int(t_final / dt)

# Function to apply periodic boundary conditions
def apply_periodic_bc(u):
    u[0] = u[-2]
    u[-1] = u[1]

# Function to compute spatial derivatives
def compute_derivatives(u, epsilon):
    dudx = np.zeros_like(u)
    d2udx2 = np.zeros_like(u)
    dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    d2udx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2)
    return -c * dudx + epsilon * d2udx2

# Solver function
def solve_advection_diffusion(epsilon):
    u = u_initial.copy()
    u_prev = u_initial.copy()
    
    # First time step using Explicit Euler
    f = compute_derivatives(u, epsilon)
    u = u + dt * f
    apply_periodic_bc(u)
    
    # Time integration using Adams-Bashforth method
    for n in range(1, N_t):
        f_prev = f
        f = compute_derivatives(u, epsilon)
        u_next = u + dt / 2 * (3 * f - f_prev)
        apply_periodic_bc(u_next)
        u_prev = u
        u = u_next
    
    return u

# Solve for undamped and damped cases
u_final_undamped = solve_advection_diffusion(epsilon_undamped)
u_final_damped = solve_advection_diffusion(epsilon_damped)

# Save the final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_final_undamped_1D_Linear_Convection_adams.npy', u_final_undamped)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_final_damped_1D_Linear_Convection_adams.npy', u_final_damped)