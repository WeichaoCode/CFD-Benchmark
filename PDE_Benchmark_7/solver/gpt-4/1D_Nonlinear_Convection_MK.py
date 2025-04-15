import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.5
dt = 0.01
T = 500
dx = dt / nu
x = np.arange(0, 2*np.pi, dx)  # spatial grid
u = np.sin(x) + 0.5*np.sin(0.5*x)  # initial condition

def maccormack(u, dx, dt, T):
    """
    The MacCormack method for solving the 1D nonlinear convection equation.
    """
    for n in range(T):
        u_star = u.copy()
        F = 0.5*u**2  # Flux
        # Predictor step
        u_star[:-1] = u[:-1] - dt/dx * (F[1:] - F[:-1])
        F_star = 0.5*u_star**2
        # Corrector step
        u[1:] = 0.5*(u[1:] + u_star[1:] - dt/dx * (F_star[1:] - F_star[:-1]))
        # Periodic boundary conditions
        u[0] = u[-1]
    return u

# Solve the PDE
u_final = maccormack(u, dx, dt, T)

# Save the solution
np.save('solution.npy', u_final)

# Plot the solution
plt.figure(figsize=(8, 5))
plt.plot(x, u_final, label='MacCormack')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of the 1D Nonlinear Convection Equation')
plt.legend()
plt.grid(True)
plt.show()