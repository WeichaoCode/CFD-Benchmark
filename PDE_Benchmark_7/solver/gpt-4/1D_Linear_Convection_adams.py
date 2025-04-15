import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Nx = 101
c = 1.0
epsilons = [0.0, 5e-4]
x = np.linspace(-5, 5, Nx)
dx = x[1] - x[0]

# Define time step based on CFL condition
dt = 0.8 * dx / c
Nt = int(1.0 / dt)

# Initialize solution and set initial condition
u = np.zeros((2, Nt, Nx))
u[:, 0, :] = np.exp(-x**2)

# Define central difference operator
def D(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2*dx)

# Time integration using Adams-Bashforth method
for epsilon_index, epsilon in enumerate(epsilons):
    for n in range(Nt-1):
        # Explicit Euler for the first step
        if n == 0:
            u[epsilon_index, n+1, :] = u[epsilon_index, n, :] - dt * c * D(u[epsilon_index, n, :], dx) + epsilon * dt * D(D(u[epsilon_index, n, :], dx), dx)
        else:
            # Adams-Bashforth for the subsequent steps
            u[epsilon_index, n+1, :] = u[epsilon_index, n, :] - dt * (3 * c * D(u[epsilon_index, n, :], dx) - c * D(u[epsilon_index, n-1, :], dx)) / 2 + epsilon * dt * (3 * D(D(u[epsilon_index, n, :], dx), dx) - D(D(u[epsilon_index, n-1, :], dx), dx)) / 2

# Save the solution
np.save('wave_profile.npy', u)

# Visualization
for epsilon_index, epsilon in enumerate(epsilons):
    plt.figure(figsize=(8, 6))
    for n in range(0, Nt, int(Nt/5)):
        plt.plot(x, u[epsilon_index, n, :], label=f't = {n*dt:.2f}')
    plt.legend()
    plt.title(f'Wave profile (epsilon = {epsilon})')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()