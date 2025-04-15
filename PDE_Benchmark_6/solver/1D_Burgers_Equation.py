import numpy as np
import matplotlib.pyplot as plt

# Variable declarations
nx = 101
nt = 100
dx = 2 * np.pi / (nx - 1)
nu = .07
dt = dx * nu

# Function to compute the derivative of phi
def dphi(x, nu, t=0):
    return -(-0.5 * x / nu * np.exp(-(x - 4 * t) ** 2 / (4 * nu * (t + 1))) - 0.5 * (x - 2 * np.pi) / nu * np.exp(-(x - 4 * t - 2 * np.pi) ** 2 / (4 * nu * (t + 1))))

# Initial condition
x = np.linspace(0, 2 * np.pi, nx)
phi = np.exp(-x ** 2 / (4 * nu)) + np.exp(-(x - 2 * np.pi) ** 2 / (4 * nu))
u = -2 * nu * dphi(x, nu) / phi + 4

# Numerical solution
un = np.empty(nx)
for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i - 1]) + nu * dt / dx ** 2 * (un[i + 1] - 2 * un[i] + un[i - 1])
    u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx ** 2 * (un[1] - 2 * un[0] + un[-2])
    u[-1] = u[0]

# Analytical solution
phi = np.exp(-(x - 4 * nt * dt) ** 2 / (4 * nu * (nt * dt + 1))) + np.exp(-(x - 4 * nt * dt - 2 * np.pi) ** 2 / (4 * nu * (nt * dt + 1)))
u_analytical = -2 * nu * dphi(x, nu, nt * dt) / phi + 4

# Plotting
plt.figure(figsize=(11, 7), dpi=100)
plt.plot(x, u, marker='o', lw=2, label='Computational')
plt.plot(x, u_analytical, label='Analytical')
plt.xlim([0, 2 * np.pi])
plt.ylim([0, 10])
plt.legend();

# Save the velocity field at last time step in a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/u_1D_Burgers_Equation.npy', u)