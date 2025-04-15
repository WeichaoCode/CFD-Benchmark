import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Nx = 101
x = np.linspace(-5, 5, Nx)
dx = x[1] - x[0]
c = 1
epsilon_values = [0, 5e-4]
dt = 0.5 * dx / c  # CFL condition
T = 1.0
Nt = int(T / dt)

# Discretize the domain
u = np.zeros((Nt, Nx, len(epsilon_values)))
u[0, :, :] = np.exp(-x**2)[None, :, None]

# Time integration using Predictor-Corrector method
for n in range(Nt - 1):
    for i, epsilon in enumerate(epsilon_values):
        # Predictor step
        u_star = np.roll(u[n, :, i], -1) - np.roll(u[n, :, i], 1)
        u_star *= -c * dt / (2 * dx)
        u_star += epsilon * dt / dx**2 * (np.roll(u[n, :, i], -1) - 2 * u[n, :, i] + np.roll(u[n, :, i], 1))
        u_star += u[n, :, i]

        # Corrector step
        u[n+1, :, i] = np.roll(u_star, -1) - np.roll(u_star, 1)
        u[n+1, :, i] *= -c * dt / (2 * dx)
        u[n+1, :, i] += epsilon * dt / dx**2 * (np.roll(u_star, -1) - 2 * u_star + np.roll(u_star, 1))
        u[n+1, :, i] += u[n, :, i]

# Save the computed solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/u_1D_Linear_Convection_pred_corr.npy', u)

# Visualization
for i, epsilon in enumerate(epsilon_values):
    plt.figure(figsize=(8, 6))
    plt.plot(x, u[0, :, i], label='Initial')
    plt.plot(x, u[Nt//2, :, i], label='Midway')
    plt.plot(x, u[-1, :, i], label='Final')
    plt.legend()
    plt.title(f'Wave profile (epsilon={epsilon})')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid(True)
    plt.show()