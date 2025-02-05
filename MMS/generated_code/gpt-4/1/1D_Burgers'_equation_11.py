import numpy as np
import matplotlib.pyplot as plt

# physical properties
nu = 0.07

# boundary conditions
u = np.zeros(2)

# initial conditions
u = np.sin(np.pi * x)

# spatial domain
x = np.linspace(0, 2, 100)

# temporal domain
T = np.linspace(0, 2, 100)

# mesh
dx = x[1] - x[0]
dt = T[1] - T[0]
r = nu * dt / dx**2

# initialize solution matrix
U = np.zeros((len(T), len(x)))
U[0, :] = u

# solve using Lax-Friedrichs method
for n in range(0, len(T)-1):
    for i in range(1, len(x)-1):
        U[n+1, i] = 0.5*(U[n, i+1] + U[n, i-1]) - dt/(2*dx) * (U[n, i+1] - U[n, i-1]) * U[n, i] + r * (U[n, i+1] - 2*U[n, i] + U[n, i-1])

# plot the solution
plt.figure(figsize=(10,6))
plt.title("1D Burgers' equation - Lax-Friedrichs method")
plt.plot(x, U[0, :], label="t = 0")
plt.plot(x, U[len(T)//4, :], label="t = T/4")
plt.plot(x, U[len(T)//2, :], label="t = T/2")
plt.plot(x, U[-1, :], label="t = T")
plt.legend()
plt.show()