import numpy as np

# Define domain
x_start = -5
x_end = 5
N_x = 101
x = np.linspace(x_start, x_end, N_x)
dx = x[1] - x[0]

# Initial condition
u0 = np.exp(-x**2)

# Parameters
c = 1
epsilons = [0, 5e-4]
save_values = ['u_undamped', 'u_damped']
t_final = 1.0

for epsilon, save_name in zip(epsilons, save_values):
    if epsilon > 0:
        dt = 0.5 * min(dx / c, dx**2 / (2 * epsilon))
    else:
        dt = 0.5 * dx / c
    n_steps = int(t_final / dt) + 1
    dt = t_final / n_steps
    u = u0.copy()
    for _ in range(n_steps):
        du_dx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        d2u_dx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        f = -c * du_dx + epsilon * d2u_dx2
        u_star = u + dt * f
        du_dx_star = (np.roll(u_star, -1) - np.roll(u_star, 1)) / (2 * dx)
        d2u_dx2_star = (np.roll(u_star, -1) - 2 * u_star + np.roll(u_star, 1)) / dx**2
        f_star = -c * du_dx_star + epsilon * d2u_dx2_star
        u = u + 0.5 * dt * (f + f_star)
    np.save(f"{save_name}.npy", u)