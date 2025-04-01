import numpy as np
import math

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
N = math.ceil(L / dx)
x = np.linspace(0, L, N, endpoint=False)
u = np.sin(x) + 0.5 * np.sin(0.5 * x)
T = 500

# Time integration using MacCormack method
for _ in range(T):
    F = 0.5 * u**2
    F_plus = np.roll(F, -1)
    u_star = u - (dt / dx) * (F_plus - F)
    F_star = 0.5 * u_star**2
    F_star_prev = np.roll(F_star, 1)
    u = 0.5 * (u + u_star - (dt / dx) * (F_star - F_star_prev))

# Save final solution
np.save('u.npy', u)