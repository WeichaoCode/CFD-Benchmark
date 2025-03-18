import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
L = 10  # length of the domain
N = 100  # number of grid points
c = 1  # convection speed
dt = 0.01  # time step
t_end = 1  # final time
epsilons = [0, 5e-4]  # damping coefficients

# Grid
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

# Initial condition
u0 = np.exp(-x**2)

# Diagonal matrix for implicit scheme
diags_values = [-epsilons[1]*dt/dx**2, 1 + 2*epsilons[1]*dt/dx**2, -epsilons[1]*dt/dx**2]
diags_indices = [-1, 0, 1]
A = diags(diags_values, diags_indices, shape=(N, N)).tocsc()

# Time integration
for epsilon in epsilons:
    u = u0.copy()
    for t in np.arange(0, t_end, dt):
        # Intermediate step
        u_star = u - c*dt/(2*dx) * (np.roll(u, -1) - np.roll(u, 1))
        # Implicit step
        b = u_star + epsilon*dt/dx**2 * (np.roll(u_star, -1) - 2*u_star + np.roll(u_star, 1))
        u = spsolve(A, b)

    # Save solution
    np.save(f'u_epsilon_{epsilon}.npy', u)

    # Plot solution
    plt.figure()
    plt.plot(x, u0, label='Initial')
    plt.plot(x, u, label='Final')
    plt.legend()
    plt.title(f'epsilon = {epsilon}')
    plt.show()