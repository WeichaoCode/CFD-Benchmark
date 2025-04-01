import numpy as np
import math

L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
N = math.ceil(L / dx)
x = np.linspace(0, L, N, endpoint=False)
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

for _ in range(500):
    F = 0.5 * u**2
    F_j_plus = np.roll(F, -1)
    F_j_minus = np.roll(F, 1)
    
    # Compute A at j+1/2 and j-1/2
    u_j_plus = np.roll(u, -1)
    u_j_minus = np.roll(u, 1)
    A_j_plus_half = 0.5 * (u + u_j_plus)
    A_j_minus_half = 0.5 * (u_j_minus + u)
    
    term1 = F_j_plus - F_j_minus
    term2 = A_j_plus_half * (F_j_plus - F) - A_j_minus_half * (F - F_j_minus)
    
    u = u - (dt / (2 * dx)) * term1 + (dt**2) / (2 * dx**2) * term2

np.save('u.npy', u)