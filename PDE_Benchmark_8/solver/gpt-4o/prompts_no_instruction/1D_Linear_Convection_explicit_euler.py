import numpy as np

# Parameters
c = 1.0
epsilon_values = [0, 5e-4]
x_start, x_end = -5, 5
Nx = 101
dx = (x_end - x_start) / (Nx - 1)
x = np.linspace(x_start, x_end, Nx)

# Initial condition
u_initial = np.exp(-x**2)

# Time-stepping parameters
CFL = 0.5
dt = CFL * dx / c
t_final = 2.0
Nt = int(t_final / dt)

# Central difference coefficients
def central_diff(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

def laplacian(u, dx):
    return (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx**2)

# Solve for each epsilon value
for epsilon in epsilon_values:
    u = u_initial.copy()
    for _ in range(Nt):
        u = u - dt * c * central_diff(u, dx) + dt * epsilon * laplacian(u, dx)
    
    # Save the final solution
    save_values = ['u_damped' if epsilon > 0 else 'u_undamped']
    np.save(save_values[0] + '.npy', u)