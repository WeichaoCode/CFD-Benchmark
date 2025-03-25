import numpy as np
import matplotlib.pyplot as plt

# Define parameters
x_min = -5.0
x_max = 5.0
Nx = 101
x = np.linspace(x_min, x_max, Nx)
dx = (x_max - x_min) / (Nx - 1)

c = 1.0  # Convection speed

# Time parameters
CFL = 0.5
epsilon_values = [0.0, 5e-4]  # Undamped and damped cases
T = 10.0
dt = 0.05
Nt = int(T / dt)

# Initial condition
u0 = np.exp(-x**2)

def compute_f(u, c, epsilon, dx):
    """
    Compute the right-hand side of the convection-diffusion equation.
    """
    du_dx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    d2u_dx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    return -c * du_dx + epsilon * d2u_dx2

# Time snapshots to save for visualization
save_times = [0, 2.5, 5.0, 7.5, 10.0]
save_steps = [int(t / dt) for t in save_times]

# Dictionary to store final solutions
final_solutions = {}

# Plot setup
plt.figure(figsize=(12, 6))

for idx, epsilon in enumerate(epsilon_values):
    u = u0.copy()
    snapshots = []
    
    for n in range(Nt + 1):
        if n in save_steps:
            snapshots.append(u.copy())
        
        # Predictor step
        f_n = compute_f(u, c, epsilon, dx)
        u_star = u + dt * f_n
        
        # Corrector step
        f_star = compute_f(u_star, c, epsilon, dx)
        u_new = u + (dt / 2) * (f_n + f_star)
        
        u = u_new.copy()
    
    final_solutions[epsilon] = u.copy()
    
    # Plotting
    for i, t in enumerate(save_times):
        plt.subplot(1, 2, idx + 1)
        plt.plot(x, snapshots[i], label=f't={t}')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title('Undamped' if epsilon == 0.0 else 'Damped')
        plt.legend()
        plt.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

# Save final solutions
np.save('final_u_undamped.npy', final_solutions[0.0])
np.save('final_u_damped.npy', final_solutions[5e-4])