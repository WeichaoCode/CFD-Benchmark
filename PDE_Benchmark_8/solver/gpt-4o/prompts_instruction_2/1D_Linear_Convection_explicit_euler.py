import numpy as np
import matplotlib.pyplot as plt

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

# Time stepping parameters
CFL = 0.5  # CFL condition number
dt = CFL * dx / c  # Time step size
t_final = 2.0  # Final time
N_t = int(t_final / dt)  # Number of time steps

def solve_wave_equation(epsilon):
    # Initialize solution
    u = u_initial.copy()
    u_new = np.zeros_like(u)

    # Time integration loop
    for n in range(N_t):
        # Apply periodic boundary conditions
        u_new[0] = u[0] - c * dt / (2 * dx) * (u[1] - u[-2]) + epsilon * dt / dx**2 * (u[1] - 2 * u[0] + u[-2])
        u_new[-1] = u_new[0]  # Periodic boundary condition

        # Update interior points
        for i in range(1, N_x - 1):
            u_new[i] = u[i] - c * dt / (2 * dx) * (u[i+1] - u[i-1]) + epsilon * dt / dx**2 * (u[i+1] - 2 * u[i] + u[i-1])

        # Update solution
        u[:] = u_new[:]

    return u

# Solve for both undamped and damped cases
u_final_undamped = solve_wave_equation(epsilon_undamped)
u_final_damped = solve_wave_equation(epsilon_damped)

# Save the final solutions to .npy files
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/u_final_undamped_1D_Linear_Convection_explicit_euler.npy', u_final_undamped)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/u_final_damped_1D_Linear_Convection_explicit_euler.npy', u_final_damped)

# Plot the final solutions for visualization
plt.figure(figsize=(10, 5))
plt.plot(x, u_final_undamped, label='Undamped', linestyle='--')
plt.plot(x, u_final_damped, label='Damped', linestyle='-')
plt.title('Wave Equation Solution at Final Time')
plt.xlabel('x')
plt.ylabel('u(x, t_final)')
plt.legend()
plt.grid(True)
plt.show()