import numpy as np
import matplotlib.pyplot as plt

# Define parameters
c = 1.0
epsilon_values = [0.0, 5e-4]
x = np.linspace(-5, 5, 101)
dx = x[1] - x[0]
dt = 0.5 * dx**2 / max(epsilon_values)  # Stability condition
t_end = 1.0
n_steps = int(t_end / dt)

# Define initial condition
u0 = np.exp(-x**2)

# Define Dufort-Frankel method
def dufort_frankel(u_prev, u_curr, r):
    u_next = np.empty_like(u_curr)
    u_next[1:-1] = ((1 - 2*r) * u_prev[1:-1] + 2*r * (u_curr[:-2] + u_curr[2:])) / (1 + 2*r)
    return u_next

# Solve for each case
for epsilon in epsilon_values:
    r = epsilon * dt / dx**2
    u_prev = u0
    u_curr = u0 - dt * c * np.gradient(u0, dx)  # First step using upwind method

    # Time integration
    for _ in range(n_steps):
        u_next = dufort_frankel(u_prev, u_curr, r)
        u_prev, u_curr = u_curr, u_next

        # Apply periodic boundary conditions
        u_curr[0] = u_curr[-2]
        u_curr[-1] = u_curr[1]

    # Save solution
    np.save(f'u_epsilon_{epsilon}.npy', u_curr)

    # Plot solution
    plt.plot(x, u_curr, label=f'epsilon = {epsilon}')

plt.legend()
plt.show()