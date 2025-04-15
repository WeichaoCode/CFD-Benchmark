import numpy as np
import matplotlib.pyplot as plt

# Define constants
Lx, Ly = 5, 4
Nx, Ny = 50, 40
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
T_top, T_bottom, T_left, T_right = 0, 20, 10, 40
tolerance = 1e-5

# Initialize grid
T = np.zeros((Ny, Nx), dtype=float)
T[0, :], T[-1, :], T[:, 0], T[:, -1] = T_top, T_bottom, T_left, T_right

# Iterative solver
def solve(T):
    error = 1
    while error > tolerance:
        T_new = T.copy()
        T_new[1:-1, 1:-1] = ((dy**2 * (T[1:-1, 2:] + T[1:-1, :-2]) +
                               dx**2 * (T[2:, 1:-1] + T[:-2, 1:-1])) /
                              (2 * (dx**2 + dy**2)))
        error = np.max(np.abs(T_new - T))
        T = T_new
    return T

# Solve
T_final = solve(T)

# Plot
plt.figure(figsize=(8, 5))
plt.title('Temperature distribution')
plt.xlabel('x')
plt.ylabel('y')
cont = plt.contourf(T_final, cmap=plt.cm.hot, levels=np.linspace(0, 40, 21))
plt.colorbar(cont)
plt.show()