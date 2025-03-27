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
u_initial = np.exp(-x**2)

# Time step based on CFL condition
CFL = 0.5
dt = CFL * dx / c
t_final = 2.0  # Final time
N_t = int(t_final / dt)

# Function to apply periodic boundary conditions
def apply_periodic_boundary(u):
    u[0] = u[-2]
    u[-1] = u[1]

# Function to compute spatial derivatives
def compute_spatial_derivatives(u, epsilon):
    dudx = np.zeros_like(u)
    d2udx2 = np.zeros_like(u)
    dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    d2udx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2)
    return -c * dudx + epsilon * d2udx2

# Function to solve the PDE
def solve_pde(epsilon):
    u = u_initial.copy()
    u_prev = u_initial.copy()
    apply_periodic_boundary(u)
    
    # First time step using Explicit Euler
    f_n = compute_spatial_derivatives(u, epsilon)
    u = u + dt * f_n
    apply_periodic_boundary(u)
    
    # Time integration using Adams-Bashforth method
    for n in range(1, N_t):
        f_n_prev = f_n
        f_n = compute_spatial_derivatives(u, epsilon)
        u_new = u + dt / 2 * (3 * f_n - f_n_prev)
        apply_periodic_boundary(u_new)
        u_prev = u
        u = u_new
    
    return u

# Solve for both undamped and damped cases
u_final_undamped = solve_pde(epsilon_undamped)
u_final_damped = solve_pde(epsilon_damped)

# Save the final solutions to .npy files
np.save('u_final_undamped.npy', u_final_undamped)
np.save('u_final_damped.npy', u_final_damped)

# Plot the results for visualization
plt.figure(figsize=(10, 5))
plt.plot(x, u_initial, label='Initial Condition', linestyle='--')
plt.plot(x, u_final_undamped, label='Final Undamped')
plt.plot(x, u_final_damped, label='Final Damped')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Wave Propagation')
plt.legend()
plt.grid(True)
plt.show()