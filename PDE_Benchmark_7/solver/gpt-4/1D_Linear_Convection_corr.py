import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Nx = 101  # Number of grid points
L = 10.0  # Width of domain
c = 1.0  # Convection speed
epsilons = [0.0, 5e-4]  # Damping factors
dx = L / (Nx - 1)  # Spatial grid size
CFL = 0.8  # CFL number for stability
dt = CFL * dx / c  # Time step size

# Discretize the domain
x = np.linspace(-L/2, L/2, Nx)

# Define the initial condition
u0 = np.exp(-x**2)

# Define the convection term
def convection(u, c, dx):
    du_dx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    return -c * du_dx

# Define the diffusion term
def diffusion(u, epsilon, dx):
    d2u_dx2 = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
    return epsilon * d2u_dx2

# Time integration using Predictor-Corrector method
for epsilon in epsilons:
    u = np.copy(u0)
    u_all = [u0]
    for n in range(int(1.0/dt)):
        f_n = convection(u, c, dx) + diffusion(u, epsilon, dx)
        u_star = u + dt * f_n
        f_star = convection(u_star, c, dx) + diffusion(u_star, epsilon, dx)
        u_new = u + 0.5 * dt * (f_n + f_star)
        u = u_new
        if n % 100 == 0:
            u_all.append(u)
    # Visualization
    for i, u in enumerate(u_all):
        plt.plot(x, u, label=f't = {i*dt:.2f}')
    plt.legend()
    plt.title(f'Wave propagation with damping factor ε = {epsilon}')
    plt.show()
    # Save the computed solution
    np.save(f'u_epsilon_{epsilon}.npy', u)