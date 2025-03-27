import numpy as np
import matplotlib.pyplot as plt

# Define parameters
c = 1.0  # Convection speed
epsilon_values = [0, 5e-4]  # Damping factors
x_start, x_end = -5, 5  # Spatial domain
N_x = 101  # Number of spatial grid points
dx = (x_end - x_start) / (N_x - 1)  # Spatial step size
x = np.linspace(x_start, x_end, N_x)  # Spatial grid

# Initial condition
u_initial = np.exp(-x**2)

# Time step determined by CFL condition
CFL = 0.5  # CFL number
dt = CFL * dx / c  # Time step size
t_final = 2.0  # Final time
N_t = int(t_final / dt)  # Number of time steps

# Function to perform predictor-corrector method
def predictor_corrector(u, epsilon):
    u_new = u.copy()
    for n in range(N_t):
        # Predictor step
        u_star = u - dt * c * (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx) + \
                 dt * epsilon * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx**2)
        
        # Corrector step
        u_new = 0.5 * (u + u_star - dt * c * (np.roll(u_star, -1) - np.roll(u_star, 1)) / (2 * dx) + \
                       dt * epsilon * (np.roll(u_star, -1) - 2 * u_star + np.roll(u_star, 1)) / (dx**2))
        
        # Update u for the next time step
        u = u_new.copy()
    
    return u_new

# Solve for both damped and undamped cases
for epsilon in epsilon_values:
    u_final = predictor_corrector(u_initial, epsilon)
    
    # Save the final solution to a .npy file
    filename = f"wave_solution_epsilon_{epsilon}.npy"
    np.save(filename, u_final)
    
    # Plot the final solution
    plt.plot(x, u_final, label=f'ε = {epsilon}')

# Plot settings
plt.title('Wave Profile at Final Time')
plt.xlabel('x')
plt.ylabel('u(x, t_final)')
plt.legend()
plt.grid(True)
plt.show()