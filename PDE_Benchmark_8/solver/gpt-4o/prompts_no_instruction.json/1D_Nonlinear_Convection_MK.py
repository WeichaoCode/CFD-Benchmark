import numpy as np

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
T = 500  # number of time steps

# Discretize the spatial domain
x = np.linspace(0, L, int(np.ceil(L / dx)), endpoint=False)
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# MacCormack method
for n in range(T):
    # Predictor step
    u_pred = np.empty_like(u)
    for i in range(nx):
        u_pred[i] = u[i] - dt * u[i] * (u[(i + 1) % nx] - u[i]) / dx

    # Corrector step
    u_corr = np.empty_like(u)
    for i in range(nx):
        u_corr[i] = 0.5 * (u[i] + u_pred[i] - dt * u_pred[i] * (u_pred[i] - u_pred[i - 1]) / dx)

    # Update solution
    u = u_corr

# Save the final solution
np.save('final_solution.npy', u)