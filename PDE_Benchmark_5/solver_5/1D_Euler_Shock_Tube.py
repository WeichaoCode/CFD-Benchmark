import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
L, T = 1.0, 0.25
nx, nt = 100, 50
gamma, CFL = 1.4, 0.5
dx, dt = L/nx, T/nt

# Space and time discretization
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Primitive variable initialization
rho, u, p = np.zeros(nx), np.zeros(nx), np.zeros(nx)
rho[x < 0.5], rho[x >= 0.5] = 1.0, 0.125
p[x < 0.5], p[x >= 0.5] = 1.0, 0.1

# Conserved variable initialization
U = np.array([rho, rho*u, p/(gamma-1) + 0.5*rho*u**2]).T
F = np.array([rho*u, rho*u**2 + p, rho*u*((gamma*p/((gamma-1)*rho) + 0.5*u**2) + p*u)]).T

# MacCormack method
U_pred = U.copy()
for n in range(nt-1):
    # Predictor step
    U_pred[:-1] = U[:-1] - dt/dx * (F[1:] - F[:-1])
    F_pred = np.array([U_pred[:,0]*U_pred[:,1], U_pred[:,0]*U_pred[:,1]**2 + U_pred[:,2]*(gamma-1), U_pred[:,0]*U_pred[:,1]*((gamma*U_pred[:, 2]/(gamma-1) + 0.5*U_pred[:,1]**2) + U_pred[:, 2]*U_pred[:,1])]).T

    # Corrector step
    U[1:] = 0.5 * (U[1:] + U_pred[1:] - dt/dx * (F_pred[1:] - F_pred[:-1]))
    F = np.array([U[:,0]*U[:,1], U[:,0]*U[:,1]**2 + U[:,2]*(gamma-1), U[:,0]*U[:,1]*((gamma*U[:, 2]/(gamma-1) + 0.5*U[:,1]**2) + U[:, 2]*U[:,1])]).T

# Convert back to primitive variables
rho, u, p = U[:,0], U[:,1]/U[:,0], (gamma-1)*(U[:,2] - 0.5*U[:,0]*(U[:,1]/U[:,0])**2)

# Plot the results
plt.figure(figsize=(12, 9))
plt.plot(x, rho, label='Density')
plt.plot(x, u, label='Velocity')
plt.plot(x, p, label='Pressure')
plt.legend(loc='best')
plt.xlabel('x')
plt.title('Density, Velocity, and Pressure profiles over time')
plt.show()