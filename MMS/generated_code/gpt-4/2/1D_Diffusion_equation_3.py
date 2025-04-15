import numpy as np
import matplotlib.pyplot as plt

# Define constants
T = 2.0
L = 2.0
nu = 1.0
N = 50
M = 50
dx = L / N
dt = T / M
alpha = nu * dt / (2.0 * dx**2)
x = np.linspace(0, L, N+1)
t = np.linspace(0, T, M+1)
u = np.zeros((M+1, N+1))

# Define initial condition
u[0, :] = np.sin(np.pi * x)

# Define boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Define the source term
def S(n, j):
    t_n = n * dt
    x_j = j * dx
    return -np.pi**2 * nu * np.exp(-t_n) * np.sin(np.pi * x_j) + np.exp(-t_n) * np.sin(np.pi * x_j)

# Solve using Crank-Nicolson method
for n in range(0, M):
    # Define the RHS of the system
    b = np.zeros(N-1)
    b[0] = alpha * u[n, 0]
    b[-1] = alpha * u[n, -1]
    b += u[n, 1:-1] + 0.5 * dt * S(n, np.arange(1, N))

    # Define the matrix A
    A = np.diag((1+2*alpha)*np.ones(N-1)) + np.diag(-alpha*np.ones(N-2), k=1) + np.diag(-alpha*np.ones(N-2), k=-1)

    # Solve the system
    u[n+1, 1:-1] = np.linalg.solve(A, b)

# Plot the solution at key time steps
plt.figure(figsize=(10, 8))
for i, n in enumerate([0, M//4, M//2, M]):
    plt.plot(x, u[n, :], label=f't = {t[n]:.2f}')
plt.title('1D Diffusion equation - Crank-Nicolson method')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()