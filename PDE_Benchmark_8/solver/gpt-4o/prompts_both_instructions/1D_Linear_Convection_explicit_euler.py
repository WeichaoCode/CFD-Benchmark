import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0  # Convection speed
epsilon_undamped = 0.0
epsilon_damped = 5e-4
x_start, x_end = -5, 5
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
    u_new = np.zeros_like(u)
    
    for n in range(N_t):
        # Central difference for spatial derivatives
        u_x = (u[2:] - u[:-2]) / (2 * dx)
        u_xx = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2)
        
        # Update using explicit Euler method
        u_new[1:-1] = u[1:-1] - c * dt * u_x + epsilon * dt * u_xx
        
        # Apply periodic boundary conditions
        apply_periodic_boundary(u_new)
        
        # Update solution
        u[:] = u_new[:]
    
    return u

# Solve for both undamped and damped cases
u_final_undamped = solve_pde(epsilon_undamped)
u_final_damped = solve_pde(epsilon_damped)

# Save the final solutions to .npy files
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_final_undamped_1D_Linear_Convection_explicit_euler.npy', u_final_undamped)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_final_damped_1D_Linear_Convection_explicit_euler.npy', u_final_damped)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x, u_initial, label='Initial Condition', linestyle='--')
plt.plot(x, u_final_undamped, label='Final Undamped')
plt.plot(x, u_final_damped, label='Final Damped')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Wave Evolution')
plt.legend()
plt.grid(True)
plt.show()