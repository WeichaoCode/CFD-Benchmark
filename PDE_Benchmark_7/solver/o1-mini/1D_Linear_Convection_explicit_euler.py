import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0
epsilon_values = [0.0, 5e-4]  # Undamped and damped cases
domain = (-5.0, 5.0)
N_x = 101
x = np.linspace(domain[0], domain[1], N_x)
dx = x[1] - x[0]
T = 2.0  # Total time
CFL_conv = 0.5
CFL_diff = 0.25

# Function to perform the simulation
def simulate(epsilon, filename, label):
    # Calculate time step based on CFL conditions
    dt_conv = CFL_conv * dx / c
    if epsilon > 0:
        dt_diff = CFL_diff * dx**2 / epsilon
        dt = min(dt_conv, dt_diff)
    else:
        dt = dt_conv
    N_t = int(T / dt) + 1
    dt = T / N_t  # Recalculate dt to exactly reach T

    # Initial condition
    u = np.exp(-x**2)
    u_initial = u.copy()

    # Time integration using Explicit Euler
    for n in range(N_t):
        # Compute derivatives with periodic boundary conditions
        u_plus = np.roll(u, -1)
        u_minus = np.roll(u, 1)
        
        du_dx = (u_plus - u_minus) / (2 * dx)
        d2u_dx2 = (u_plus - 2 * u + u_minus) / dx**2
        
        # Update u
        u_new = u - dt * c * du_dx + dt * epsilon * d2u_dx2
        u = u_new

    # Save the final solution
    np.save(filename, u)

    # Plotting
    plt.plot(x, u_initial, 'k--', label='Initial')
    plt.plot(x, u, label=label)

# Plot setup
plt.figure(figsize=(10, 6))

# Simulate both cases
simulate(epsilon=0.0, filename='u_final_undamped.npy', label='Undamped (ε=0)')
simulate(epsilon=5e-4, filename='u_final_damped.npy', label='Damped (ε=5e-4)')

# Final plot adjustments
plt.xlabel('x')
plt.ylabel('u(x, T)')
plt.title('Final Wave Profiles at t = {:.2f}'.format(T))
plt.legend()
plt.grid(True)
plt.savefig('final_wave_profiles.png')
plt.show()