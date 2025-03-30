import numpy as np
import math

# Parameters
L = 2.0 * math.pi
nu = 0.5
dt = 0.01
dx = dt / nu  # 0.02
n_points = math.ceil(L / dx)  # 315
x = np.linspace(0, L, n_points, endpoint=False)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Number of time steps
T = 500

for _ in range(T):
    # Predictor step: forward difference
    du_dx_forward = (np.roll(u, -1) - u) / dx
    u_pred = u - dt * u * du_dx_forward

    # Corrector step: backward difference
    du_dx_backward = (u_pred - np.roll(u_pred, 1)) / dx
    u_new = 0.5 * (u + u_pred - dt * u_pred * du_dx_backward)

    u = u_new

# Save the final solution
np.save('u.npy', u)