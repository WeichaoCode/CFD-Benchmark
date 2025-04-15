import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Nx = 101
c = 1
epsilons = [0, 5e-4]
x = np.linspace(-5, 5, Nx)
dx = x[1] - x[0]
dt = 0.5 * dx / c  # CFL condition

# Define the function for the right-hand side of the ODE
def f(u, epsilon):
    du_dx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    d2u_dx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    return -c * du_dx + epsilon * d2u_dx2

# Initialize variables
u0 = np.exp(-x**2)

# Time integration using Runge-Kutta method
for epsilon in epsilons:
    u = u0.copy()
    for n in range(1000):
        k1 = f(u, epsilon)
        k2 = f(u + dt/2 * k1, epsilon)
        k3 = f(u + dt/2 * k2, epsilon)
        k4 = f(u + dt * k3, epsilon)
        u += dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    # Visualization
    plt.plot(x, u, label=f'epsilon = {epsilon}')

plt.legend()
plt.show()

# Save the computed solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/u_1D_Linear_Convection_rk.npy', u)