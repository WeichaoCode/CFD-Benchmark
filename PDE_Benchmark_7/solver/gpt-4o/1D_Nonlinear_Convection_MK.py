import numpy as np
import matplotlib.pyplot as plt
import math

# Define parameters
nu = 0.5  # CFL number
dt = 0.01
T = 500
dx = dt / nu
L = 2 * np.pi
nx = math.ceil(L / dx)
x = np.linspace(0, L, nx, endpoint=False)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Function to compute flux
def flux(u):
    return 0.5 * u**2

def macCormack(u, dx, dt):
    # Predictor step
    f = flux(u)
    u_star = u.copy()
    for j in range(nx):
        u_star[j] = u[j] - (dt / dx) * (f[(j+1) % nx] - f[j])

    # Corrector step
    f_star = flux(u_star)
    u_new = u.copy()
    for j in range(nx):
        u_new[j] = 0.5 * (u[j] + u_star[j]
                          - (dt / dx) * (f_star[j] - f_star[j-1]))

    return u_new

# Run the simulation
for n in range(T):
    u = macCormack(u, dx, dt)

# Save the final result
np.save('/PDE_Benchmark_7/results/prediction/u_1D_Nonlinear_Convection_Mk.npy', u)

# Comparison plotting (not implemented here for Lax and Lax-Wendroff)
plt.plot(x, u, label='MacCormack', color='r')
plt.title('1D Nonlinear Convection using MacCormack method')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()