import numpy as np
import matplotlib.pyplot as plt

# Define the initial parameters
gamma = 1.4  # ratio of specific heats
L = 2.  # tube length
nx = 81  # number of nodes
dx = L / (nx - 1)  # spatial step size
T = 0.25  # final time
CFL = 1.0  # Courant-Friedrichs-Lewy condition
dt = (CFL / nx) * L  # temporal step size

# Define the initial condition parameters
rho_L = 1.0  # left density
u_L = 0.0  # left velocity
p_L = 1.0  # left pressure
rho_R = 0.125  # right density
u_R = 0.0  # right velocity
p_R = 0.1  # right pressure

# Discretise the domain
x = np.linspace(-1, 1, nx)

# Calculate the initial values of U and F
U = np.zeros([3, nx])
F = np.zeros([3, nx])

# Set up the initial density, velocity, and pressure values
U[0, :int(nx / 2)] = rho_L
U[0, int(nx / 2):] = rho_R
U[1, :int(nx / 2)] = rho_L * u_L
U[1, int(nx / 2):] = rho_R * u_R
U[2, :int(nx / 2)] = p_L / (gamma - 1) + 0.5 * rho_L * (u_L ** 2)
U[2, int(nx / 2):] = p_R / (gamma - 1) + 0.5 * rho_R * (u_R ** 2)

# Main time-stepping loop
for n in range(int(T / dt)):
    F[0, :] = U[1, :]
    F[1, :] = ((U[1, :] ** 2) / U[0, :]) + (gamma - 1) * (U[2, :] - 0.5 * (U[1, :] ** 2) / U[0, :])
    F[2, :] = (U[2, :] + (gamma - 1) * (U[2, :] - 0.5 * (U[1, :] ** 2) / U[0, :])) * (U[1, :] / U[0, :])

    U_star = np.zeros_like(U)
    F_star = np.zeros_like(F)

    # Predictor step
    U_star[:, :-1] = U[:, :-1] - dt / dx * (F[:, 1:] - F[:, :-1])

    # Corrector step
    F_star[0, :] = U_star[1, :]
    F_star[1, :] = ((U_star[1, :] ** 2) / U_star[0, :]) + (gamma - 1) * (U_star[2, :] - 0.5 * (U_star[1, :] ** 2) / U_star[0, :])
    F_star[2, :] = (U_star[2, :] + (gamma - 1) * (U_star[2, :] - 0.5 * (U_star[1, :] ** 2) / U_star[0, :])) * (U_star[1, :] / U_star[0, :])

    U[:, 1:-1] = 0.5 * (U[:, 1:-1] + U_star[:, 1:-1] - dt / dx * (F_star[:, 1:-1] - F_star[:, :-2]))

# Extract the computed density, velocity, and pressure profiles
rho = U[0, :]
u = U[1, :] / rho
p = (gamma - 1) * (U[2, :] - 0.5 * rho * u ** 2)

# Create plots
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(x, rho, 'k-')
plt.title('Density')

plt.subplot(132)
plt.plot(x, u, 'k-')
plt.title('Velocity')

plt.subplot(133)
plt.plot(x, p, 'k-')
plt.title('Pressure')

plt.tight_layout()
plt.show()

# Save the computed density, velocity, and pressure profiles
np.save('rho.npy', rho)
np.save('u.npy', u)
np.save('p.npy', p)

# Save U, F
np.save('U.npy', U)
np.save('F.npy', F)