import numpy as np
import matplotlib.pyplot as plt

# Define domain
x_start = -5.0
x_end = 5.0
N_x = 101
x = np.linspace(x_start, x_end, N_x)
dx = x[1] - x[0]

# Initial condition
u_initial = np.exp(-x**2)

# Time parameters
T = 1.0
c = 1.0
epsilon_list = [0, 5e-4]

# Initialize storage for final solutions
final_solutions = {}

for epsilon in epsilon_list:
    if epsilon == 0:
        dt = dx / c * 0.5
    else:
        dt_advect = dx / c
        dt_diffuse = dx**2 / (2 * epsilon)
        dt = 0.4 * min(dt_advect, dt_diffuse)
    
    n_steps = int(T / dt)
    u = u_initial.copy()
    
    for _ in range(n_steps):
        u_p = np.roll(u, 1)
        u_m = np.roll(u, -1)
        du_dx = (u_m - u_p) / (2 * dx)
        d2u_dx2 = (u_m - 2 * u + u_p) / (dx**2)
        u_new = u - c * dt * du_dx + epsilon * dt * d2u_dx2
        u = u_new
    
    if epsilon == 0:
        final_solutions['u'] = u.copy()
        np.save('u.npy', final_solutions['u'])
    else:
        final_solutions['u_damped'] = u.copy()
        np.save('u_damped.npy', final_solutions['u_damped'])

# Visualization
plt.figure(figsize=(8, 6))
if 'u' in final_solutions:
    plt.plot(x, final_solutions['u'], label='ε=0')
if 'u_damped' in final_solutions:
    plt.plot(x, final_solutions['u_damped'], label='ε=5e-4')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Wave Evolution at Final Time Step')
plt.legend()
plt.grid(True)
plt.show()