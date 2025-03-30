import numpy as np

# Define domain
x_start = -5.0
x_end = 5.0
N_x = 101
x = np.linspace(x_start, x_end, N_x)
dx = (x_end - x_start) / (N_x - 1)

# Initial condition
u0 = np.exp(-x**2)

# Parameters
c = 1.0
epsilons = [0.0, 5e-4]
t_max = 1.0

for epsilon in epsilons:
    if epsilon > 0:
        dt = 0.9 * min(dx / c, dx**2 / (2 * epsilon))
    else:
        dt = 0.9 * dx / c
    n_steps = int(t_max / dt) + 1
    dt = t_max / n_steps
    u = u0.copy()
    for _ in range(n_steps):
        u_plus = np.roll(u, -1)
        u_minus = np.roll(u, 1)
        du_dx = (u_plus - u_minus) / (2 * dx)
        d2u_dx2 = (u_plus - 2 * u + u_minus) / dx**2
        u = u + dt * (-c * du_dx + epsilon * d2u_dx2)
    if epsilon == 0.0:
        np.save('u_undamped.npy', u)
    else:
        np.save('u_damped.npy', u)