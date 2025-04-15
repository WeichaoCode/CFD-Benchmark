"""
1D linear convection equation
---------------------------------------------------------------
- Exercise 4 - Finite difference solution of the 1D linear convection equation
- Explicit Euler method
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""
import numpy as np
import matplotlib.pyplot as plt

nx = 101
xmin = -5
xmax = 5
x = np.linspace(xmin, xmax, nx)
dx = (xmax - xmin) / nx
c = 1


def func(u, epsilon):
    # Sub-routine to evaluate the RHS function. Uses 2nd-order centered differences to discretize the spatial
    # derivatives
    f = 0 * u.copy()
    f[0] = epsilon * (u[1] - 2 * u[0] + u[-2]) / (dt ** 2) - (u[1] - u[-2]) / (2 * dx)
    f[1:-1] = epsilon * (u[2:] - 2 * u[1:-1] + u[:-2]) / (dt ** 2) - (u[2:] - u[:-2]) / (2 * dx)
    f[-1] = epsilon * (u[1] - 2 * u[-1] + u[-2]) / (dt ** 2) - (u[1] - u[-2]) / (2 * dx)
    return f


def L2norm(u):
    # Sub-routine to evaluate the L2 norm of the error
    return (np.sum((u - np.exp(-x ** 2)) ** 2)) ** 0.5


# Explicit Euler method
# Initial condition
u = np.exp(-x ** 2)

# Define time step
CFL = 0.2
dt = CFL * dx / c

## Undamped case
epsilon = 0

t = 0
tmax = 10  # This is the time required for the wave to convect back to its original position
plt.plot(x, u, label='$t=0$')
while t < tmax:
    un = u.copy()
    u = un + dt * func(un, epsilon)
    t += dt
plt.plot(x, u, label='$\epsilon = 0$')
plt.title('Explicit Euler')
plt.grid()
plt.xlabel('$x$')
plt.ylabel('$u$')

print('Undamped L2norm(error) =', L2norm(u))

## Damped case
epsilon = 5e-4

# Initial condition
u = np.exp(-x ** 2)

t = 0
while t < tmax:
    un = u.copy()
    u = un + dt * func(un, epsilon)
    t += dt
plt.plot(x, u, label='$\epsilon = 5 \\times 10^{-4}$')
plt.legend()

print('Damped L2norm(error) =', L2norm(u))
plt.show()

##########################################################################
import os

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "results")

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# Get the current Python file name without extension
python_filename = os.path.splitext(os.path.basename(__file__))[0]

# Define the file name dynamically
output_file = os.path.join(OUTPUT_FOLDER, f"u_{python_filename}.npy")

# Save the array u in the results folder
np.save(output_file, u)
