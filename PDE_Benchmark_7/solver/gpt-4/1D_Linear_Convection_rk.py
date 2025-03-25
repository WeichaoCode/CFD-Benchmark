import numpy as np
import matplotlib.pyplot as plt

# Define the PDE
def pde(u, c, dx, eps):
    du_dx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    d2u_dx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    return -c * du_dx + eps * d2u_dx2

# Define the Runge-Kutta method
def rk4(u, dt, c, dx, eps):
    k1 = pde(u, c, dx, eps)
    k2 = pde(u + dt/2 * k1, c, dx, eps)
    k3 = pde(u + dt/2 * k2, c, dx, eps)
    k4 = pde(u + dt * k3, c, dx, eps)
    return u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Define the parameters
Nx = 101
x = np.linspace(-5, 5, Nx)
dx = x[1] - x[0]
c = 1
eps = [0, 5e-4]
dt = 0.5 * dx / c  # CFL condition

# Initialize the solution
u0 = np.exp(-x**2)
u = np.zeros((2, Nx))
u[0, :] = u0
u[1, :] = u0

# Time integration
Nt = 1000
for i in range(Nt):
    for j in range(2):
        u[j, :] = rk4(u[j, :], dt, c, dx, eps[j])

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(x, u[0, :], label='Undamped')
plt.plot(x, u[1, :], label='Damped')
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.title('Wave profile at t = {}'.format(Nt*dt))
plt.grid(True)
plt.show()

# Save the solution
np.save('solution.npy', u)