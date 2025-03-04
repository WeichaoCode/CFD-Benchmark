import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
L = 1.0  # Length of domain
N = 100  # Number of grid points
T = 1.0  # Time at which to compute the solution
dt = 0.001  # Time step size
gamma = 1.4  # Ratio of specific heats

# Grid
x = np.linspace(0, L, N)

# Initial conditions
rho0 = np.exp(-0) * np.sin(np.pi * x)
u0 = np.exp(-0) * np.cos(np.pi * x)
p0 = np.exp(-0) * (1 + np.sin(np.pi * x))
E0 = p0 / (gamma - 1) + 0.5 * rho0 * u0**2

# Pack initial conditions into a single array
U0 = np.array([rho0, rho0 * u0, E0])

# Source terms
def S(U, x):
    rho, rhou, E = U
    u = rhou / rho
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    return np.array([
        -rho * np.pi * np.cos(np.pi * x),
        -u * np.pi * np.cos(np.pi * x) - 2 * p * np.pi * np.sin(np.pi * x),
        -u * np.pi * np.cos(np.pi * x) * (E + p)
    ])

# Flux vector
def F(U):
    rho, rhou, E = U
    u = rhou / rho
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    return np.array([rhou, rhou * u + p, (E + p) * u])

# Euler equations
def dUdt(U, x):
    return -np.gradient(F(U), x, edge_order=2) + S(U, x)

# Solve the PDE
U = odeint(dUdt, U0, x)

# Extract the solution
rho, rhou, E = U.T
u = rhou / rho
p = (gamma - 1) * (E - 0.5 * rho * u**2)

# Exact solution
rho_exact = np.exp(-T) * np.sin(np.pi * x)
u_exact = np.exp(-T) * np.cos(np.pi * x)
p_exact = np.exp(-T) * (1 + np.sin(np.pi * x))

# Plot the solution
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x, rho, label='Numerical')
plt.plot(x, rho_exact, label='Exact')
plt.title('Density')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(x, u, label='Numerical')
plt.plot(x, u_exact, label='Exact')
plt.title('Velocity')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(x, p, label='Numerical')
plt.plot(x, p_exact, label='Exact')
plt.title('Pressure')
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(x, np.abs(rho - rho_exact), label='Density')
plt.plot(x, np.abs(u - u_exact), label='Velocity')
plt.plot(x, np.abs(p - p_exact), label='Pressure')
plt.title('Absolute Error')
plt.legend()
plt.tight_layout()
plt.show()