import numpy as np

# Parameters
c = 1.0  # Convection speed
epsilon = 5e-4  # Damping factor
x_start, x_end = -5, 5  # Spatial domain
N_x = 101  # Number of spatial grid points
dx = (x_end - x_start) / (N_x - 1)  # Spatial step size
x = np.linspace(x_start, x_end, N_x)  # Spatial grid

# Initial condition
u_initial = np.exp(-x**2)

# Time step based on CFL condition
CFL = 0.5  # CFL number for stability
dt = CFL * dx / c  # Time step size

# Total time and number of time steps
T_final = 2.0  # Final time
N_t = int(T_final / dt)  # Number of time steps

# Initialize solution
u = u_initial.copy()

# Time integration using Explicit Euler method
for n in range(N_t):
    # Compute spatial derivatives using central differences
    u_x = np.roll(u, -1) - np.roll(u, 1)
    u_xx = np.roll(u, -1) - 2 * u + np.roll(u, 1)
    
    # Update solution
    u_new = u - c * (dt / (2 * dx)) * u_x + epsilon * (dt / dx**2) * u_xx
    
    # Apply periodic boundary conditions
    u_new[0] = u_new[-2]
    u_new[-1] = u_new[1]
    
    # Update for next time step
    u = u_new

# Save the final solution as a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_1/u_1D_Linear_Convection_explicit_euler.npy', u)