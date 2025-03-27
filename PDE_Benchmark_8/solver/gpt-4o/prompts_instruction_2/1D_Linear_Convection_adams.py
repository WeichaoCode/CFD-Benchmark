import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0  # Convection speed
epsilon_values = [0, 5e-4]  # Damping factors
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

# Function to apply periodic boundary conditions
def apply_periodic_bc(u):
    u[0] = u[-2]
    u[-1] = u[1]

# Function to compute the solution
def solve_pde(epsilon):
    u = u_initial.copy()
    u_new = np.zeros_like(u)
    
    # First time step using Explicit Euler
    for n in range(1, N_t + 1):
        # Compute spatial derivatives
        u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx**2)
        
        # Explicit Euler for the first step
        if n == 1:
            u_new = u - dt * (c * u_x - epsilon * u_xx)
        else:
            # Adams-Bashforth method for subsequent steps
            u_new = u - dt * (1.5 * (c * u_x - epsilon * u_xx) - 0.5 * (c * u_x_prev - epsilon * u_xx_prev))
        
        # Apply periodic boundary conditions
        apply_periodic_bc(u_new)
        
        # Update previous derivatives
        u_x_prev = u_x.copy()
        u_xx_prev = u_xx.copy()
        
        # Update solution
        u = u_new.copy()
    
    return u

# Solve for both undamped and damped cases
for epsilon in epsilon_values:
    final_solution = solve_pde(epsilon)
    filename = f"final_solution_epsilon_{epsilon}.npy"
    np.save(filename, final_solution)
    print(f"Final solution saved to {filename}")

    # Plot the final solution
    plt.plot(x, final_solution, label=f'ε = {epsilon}')
    
plt.title('Final Solution at t = 2.0')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.grid(True)
plt.show()