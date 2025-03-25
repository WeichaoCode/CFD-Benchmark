import numpy as np
import matplotlib.pyplot as plt

# Define the domain
Lx = 5.0
Ly = 4.0
nx = int(Lx / 0.05) + 1
ny = int(Ly / 0.05) + 1
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
beta = dx / dy

# Initialize the solution
T = np.zeros((ny, nx))

# Set the boundary conditions
T[:, 0] = 10.0  # AB
T[0, :] = 0.0   # CD
T[:, -1] = 40.0 # EF
T[-1, :] = 20.0 # GH

# Define the SOR solver
def sor_solver(T, omega, beta, nx, ny, iter_max=20000, tol=1e-6):
    T_new = np.copy(T)
    iter_count = 0
    residual = 1e10  # initial residual
    while iter_count < iter_max and residual > tol:
        T_old = np.copy(T_new)
        for j in range(1, ny - 1):  # loop over y (row)
            for i in range(1, nx - 1):  # loop over x (column)
                T_new[j, i] = (omega / (2.0 * (1.0 + beta**2))) * ((T_new[j, i-1] + T_old[j, i+1]) + beta**2 * (T_new[j-1, i] + T_old[j+1, i])) + (1.0 - omega) * T_old[j, i]
        residual = np.linalg.norm(T_new - T_old)
        iter_count += 1
    return T_new

# Solve the problem
omega = 1.5  # relaxation factor
T = sor_solver(T, omega, beta, nx, ny)

# Save the solution to a file
np.save('temperature.npy', T)

# Plot the solution
plt.figure(figsize=(8, 5))
plt.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), T, levels=50, cmap='jet')
plt.title('Steady-State Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Temperature (C)')
plt.show()