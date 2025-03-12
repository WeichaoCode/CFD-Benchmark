import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 1.0  # length of the shock tube
T = 0.2  # time interval
nx = 101  # number of spatial discrete points
nt = 81  # number of temporal discrete points
gamma = 1.4  # heat capacity ratio 
CFL = 0.6  # Courant–Friedrichs–Lewy condition

dx = L / (nx - 1)  # distance between two discrete points in space
dt = CFL * dx  # time step size

# Initialize arrays
U = np.zeros((3, nx))  # array to store the solution
U_new = U.copy()

# Initial conditions
U[:, :nx//2] = np.array([1, 0, 1 / (gamma - 1)])[:, None]  # left region
U[:, nx//2:] = np.array([0.125, 0, 0.1 / (gamma - 1)])[:, None]  # right region

# MacCormack Method
for _ in range(nt):
    F = np.array([U[1], U[1]**2 / U[0] + (gamma - 1) * (U[2] - 0.5 * U[1]**2 / U[0]), U[1] * (U[2] + (gamma - 1) * (U[2] - 0.5 * U[1]**2 / U[0])) / U[0]])

    U_new[:, :-1] = U[:, :-1] - dt / dx * (F[:, 1:] - F[:, :-1])
    U_new[:, -1] = U[:, -1]

    F_new = np.array([U_new[1], U_new[1]**2 / U_new[0] + (gamma - 1) * (U_new[2] - 0.5 * U_new[1]**2 / U_new[0]), U_new[1] * (U_new[2] + (gamma - 1) * (U_new[2] - 0.5 * U_new[1]**2 / U_new[0])) / U_new[0]])

    U[:, 1:] = 0.5 * (U[:, 1:] + U_new[:, 1:] - dt / dx * (F_new[:, 1:] - F_new[:, :-1]))

# Plot results
x = np.linspace(0, L, nx)
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.plot(x, U[0])
plt.title('Density')
plt.subplot(132)
plt.plot(x, U[1] / U[0])
plt.title('Velocity')
plt.subplot(133)
plt.plot(x, (gamma - 1) * (U[2] - 0.5 * U[1]**2 / U[0]))
plt.title('Pressure')
plt.show()