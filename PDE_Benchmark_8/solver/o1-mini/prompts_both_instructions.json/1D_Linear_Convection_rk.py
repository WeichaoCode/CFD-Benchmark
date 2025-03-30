import numpy as np
import matplotlib.pyplot as plt

def compute_rhs(u, c, epsilon, dx):
    # Apply periodic boundary conditions
    u_p = np.roll(u, -1)
    u_m = np.roll(u, 1)
    du_dx = (u_p - u_m) / (2 * dx)
    d2u_dx2 = (u_p - 2 * u + u_m) / dx**2
    return -c * du_dx + epsilon * d2u_dx2

def rk4_step(u, dt, c, epsilon, dx):
    k1 = compute_rhs(u, c, epsilon, dx)
    k2 = compute_rhs(u + 0.5 * dt * k1, c, epsilon, dx)
    k3 = compute_rhs(u + 0.5 * dt * k2, c, epsilon, dx)
    k4 = compute_rhs(u + dt * k3, c, epsilon, dx)
    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Domain parameters
x_start = -5.0
x_end = 5.0
N_x = 101
x = np.linspace(x_start, x_end, N_x)
dx = (x_end - x_start) / (N_x - 1)

# Initial condition
u_initial = np.exp(-x**2)

# Parameters
c = 1.0
epsilon_values = [0.0, 5e-4]
t_final = 1.0

for epsilon in epsilon_values:
    if epsilon == 0.0:
        dt = 0.5 * dx / c
        filename = 'u_undamped.npy'
    else:
        dt_conv = dx / c
        dt_diff = dx**2 / (2 * epsilon)
        dt = 0.5 * min(dt_conv, dt_diff)
        filename = 'u_damped.npy'
    
    N_t = int(t_final / dt)
    dt = t_final / N_t  # Adjust dt to reach t_final exactly
    u = u_initial.copy()
    
    for _ in range(N_t):
        u = rk4_step(u, dt, c, epsilon, dx)
    
    np.save(filename, u)
    plt.plot(x, u, label=f'ε = {epsilon}')

plt.xlabel('x')
plt.ylabel('u(x, t_final)')
plt.title('Wave Propagation at Final Time Step')
plt.legend()
plt.show()