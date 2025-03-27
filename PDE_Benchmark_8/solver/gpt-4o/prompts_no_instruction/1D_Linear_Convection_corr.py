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
CFL = 0.5  # CFL number
dt = CFL * dx / c  # Time step size
t_final = 2.0  # Final time
N_t = int(t_final / dt)  # Number of time steps

# Initialize solution
u = u_initial.copy()

# Predictor-Corrector method
for n in range(N_t):
    # Predictor step
    u_pred = u.copy()
    for i in range(1, N_x - 1):
        u_pred[i] = u[i] - dt * c * (u[i+1] - u[i-1]) / (2 * dx) + \
                    dt * epsilon * (u[i+1] - 2*u[i] + u[i-1]) / (dx**2)
    
    # Periodic boundary conditions
    u_pred[0] = u_pred[-2]
    u_pred[-1] = u_pred[1]
    
    # Corrector step
    for i in range(1, N_x - 1):
        u[i] = 0.5 * (u[i] + u_pred[i] - dt * c * (u_pred[i+1] - u_pred[i-1]) / (2 * dx) + \
                      dt * epsilon * (u_pred[i+1] - 2*u_pred[i] + u_pred[i-1]) / (dx**2))
    
    # Periodic boundary conditions
    u[0] = u[-2]
    u[-1] = u[1]

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_1D_Linear_Convection_corr.npy', u)