import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0  # Convection speed
epsilon = 5e-4  # Damping factor
x_start, x_end = -5, 5
N_x = 101
dx = (x_end - x_start) / (N_x - 1)
x = np.linspace(x_start, x_end, N_x)

# Initial condition
u_initial = np.exp(-x**2)

# Time step based on CFL condition
CFL = 0.5
dt = CFL * dx / c

# Time integration parameters
t_final = 2.0
n_steps = int(t_final / dt)

# Function to compute spatial derivatives
def compute_derivatives(u, dx, epsilon):
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

# Time integration using predictor-corrector method
def solve_pde(u_initial, dt, n_steps, dx, epsilon):
    u = u_initial.copy()
    for _ in range(n_steps):
        # Predictor step
        f_n = compute_derivatives(u, dx, epsilon)
        u_star = u + dt * f_n

        # Corrector step
        f_star = compute_derivatives(u_star, dx, epsilon)
        u = u + (dt / 2) * (f_n + f_star)

    return u

# Solve for both damped and undamped cases
u_final_damped = solve_pde(u_initial, dt, n_steps, dx, epsilon)
u_final_undamped = solve_pde(u_initial, dt, n_steps, dx, 0)

# Save the final solution to a .npy file
np.save('final_solution_damped.npy', u_final_damped)
np.save('final_solution_undamped.npy', u_final_undamped)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, u_initial, label='Initial Condition', linestyle='--')
plt.plot(x, u_final_damped, label='Final Solution (Damped)')
plt.plot(x, u_final_undamped, label='Final Solution (Undamped)')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Wave Propagation with Damping')
plt.legend()
plt.grid(True)
plt.show()