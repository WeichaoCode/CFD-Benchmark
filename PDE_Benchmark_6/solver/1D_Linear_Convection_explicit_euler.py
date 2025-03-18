import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
c = 1.0
epsilon = [0.0, 5e-4]
x_start, x_end = -5.0, 5.0
nx = 101
dx = (x_end - x_start) / (nx - 1)
dt = dx / c
nt = 1001
x = np.linspace(x_start, x_end, nx)

# Define the initial condition
u0 = np.exp(-x**2)

# Define the explicit method
def explicit_method(u, epsilon):
    for n in range(nt - 1):
        un = u.copy()
        u[1:-1] = (un[1:-1] - c * dt / dx * (un[1:-1] - un[:-2]) +
                   epsilon * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2]))
        u[0] = u[-2]  # Apply periodic boundary conditions
        u[-1] = u[1]  # Apply periodic boundary conditions
    return u

# Solve the equation for each case
for eps in epsilon:
    u = u0.copy()
    u = explicit_method(u, eps)
    plt.plot(x, u, label=f'epsilon = {eps}')

# Plot the solution
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of the 1D linear convection equation')
plt.grid(True)
plt.show()

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/u_1D_Linear_Convection_explicit_euler.npy', u)