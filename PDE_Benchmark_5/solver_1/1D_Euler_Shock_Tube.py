import numpy as np
import matplotlib.pyplot as plt

# Define Parameters
L = 1.0  # Length of the tube
T = 0.2  # Time at which we want to get the solution
nx = 201  # No. of grid points
nt = 1500  # No. of time steps
gamma = 1.4  # Ratio of specific heats
CFL = 0.5  # Courant–Friedrichs–Lewy conditions

# Discretize spatial and time 
dx = L / (nx - 1)
dt = T / (nt - 1)
x = np.linspace(0, L, nx)

# Initialize primitive and conservative variables
U = np.zeros((3, nx))
U_new = np.zeros_like(U)
U_star = np.zeros_like(U)
F = np.zeros_like(U)

# Initial Conditions
U[0, :int(0.5 / dx)] = 1
U[0, int(0.5 / dx):] = 0.125
U[1, :] = 0
U[2, :int(0.5 / dx)] = 2.5 / (gamma - 1)
U[2, int(0.5 / dx):] = 0.25 / (gamma - 1)

# MacCormack scheme
for t in range(nt):
    F[0, :] = U[1, :]
    F[1, :] = ((U[1, :] ** 2) / U[0, :]) + (gamma - 1) * (U[2, :] - 0.5 * (U[1, :] ** 2) / U[0, :])
    F[2, :] = (U[2, :] + (gamma - 1) * (U[2, :] - 0.5 * (U[1, :] ** 2) / U[0, :])) * (U[1, :] / U[0, :])
    
    U_star[:, :-1] = U[:, :-1] - dt / dx * (F[:, 1:] - F[:, :-1])
    U_star[:, -1] = U[:, -1]
    
    F_star = np.zeros_like(U)
    F_star[0, :] = U_star[1, :]
    F_star[1, :] = ((U_star[1, :] ** 2) / U_star[0, :]) + (gamma - 1) * (U_star[2, :] - 0.5 * (U_star[1, :] ** 2) / U_star[0, :])
    F_star[2, :] = (U_star[2, :] + (gamma - 1) * (U_star[2, :] - 0.5 * (U_star[1, :] ** 2) / U_star[0, :])) * (U_star[1, :] / U_star[0, :])
    
    U_new[:, 1:] = 0.5 * (U[:, 1:] + U_star[:, 1:] - dt / dx * (F_star[:, 1:] - F_star[:, :-1]))
    U_new[:, 0] = U[:, 0]
    
    U = U_new.copy()

# Convert back to primitive variables and plot
rho = U[0, :]
u = U[1, :] / rho
p = (gamma - 1) * (U[2, :] - 0.5 * rho * u ** 2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(x, rho, 'k-')
plt.title('Density')

plt.subplot(1, 3, 2)
plt.plot(x, u, 'k-')
plt.title('Velocity')

plt.subplot(1, 3, 3)
plt.plot(x, p, 'k-')
plt.title('Pressure')

plt.tight_layout()
plt.show()