import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0
epsilon_values = [0.0, 5e-4]
x_start = -5.0
x_end = 5.0
N_x = 101
t_final = 2.0
save_files = ['u_eps0.npy', 'u_eps5e_4.npy']

# Spatial grid
x = np.linspace(x_start, x_end, N_x)
dx = (x_end - x_start) / (N_x - 1)

# Initial condition
u_initial = np.exp(-x**2)

# Function to compute spatial derivatives with periodic boundary conditions
def compute_derivatives(u, dx):
    du_dx = np.zeros_like(u)
    d2u_dx2 = np.zeros_like(u)
    du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    du_dx[0] = (u[1] - u[-1]) / (2 * dx)
    du_dx[-1] = (u[0] - u[-2]) / (2 * dx)
    
    d2u_dx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    d2u_dx2[0] = (u[1] - 2 * u[0] + u[-1]) / dx**2
    d2u_dx2[-1] = (u[0] - 2 * u[-1] + u[-2]) / dx**2
    return du_dx, d2u_dx2

# Time integration for each epsilon
for idx, epsilon in enumerate(epsilon_values):
    u = u_initial.copy()
    
    # Determine time step based on CFL condition
    if epsilon > 0:
        dt_conv = dx / c
        dt_diff = dx**2 / (2 * epsilon)
        dt = min(dt_conv, dt_diff)
    else:
        dt = dx / c
    Nt = int(t_final / dt) + 1
    dt = t_final / Nt
    
    for n in range(Nt):
        # Predictor step
        du_dx, d2u_dx2 = compute_derivatives(u, dx)
        f_n = -c * du_dx + epsilon * d2u_dx2
        u_star = u + dt * f_n
        
        # Compute f at u_star
        du_dx_star, d2u_dx2_star = compute_derivatives(u_star, dx)
        f_star = -c * du_dx_star + epsilon * d2u_dx2_star
        
        # Corrector step
        u = u + (dt / 2) * (f_n + f_star)
    
    # Save the final solution
    np.save(save_files[idx], u)

    # Plotting
    plt.plot(x, u_initial, label='Initial')
    plt.plot(x, u, label=f'Final ε={epsilon}')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Wave Profile for ε={epsilon}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'wave_profile_eps{str(epsilon).replace(".", "e")}.png')
    plt.clf()