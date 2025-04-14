"""
2D Unsteady Heat Equation
---------------------------------------------------------------
- Exercise 3 - Unsteady two-dimensional heat equation
- This code will use Dufort-Frankel method
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/12
"""

import numpy as np
import time
import matplotlib.pyplot as plt


# Sub-routine that defines the source term:
def Q(x, y):
    sigma = 0.1
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


# Initial and boundary conditions
def ICBC(X, Y, nx, ny, T0, Q0):
    q = Q0 * Q(X, Y)
    T = T0 * np.ones((ny, nx)) + q
    T[:, 0] = T0
    T[:, nx - 1] = T0
    T[0, :] = T0
    T[ny - 1, :] = T0
    return T


# DuFort-Frankel method
def DF(T, q, alpha, beta, dx, tmax, r):
    r = r / (1 + beta ** 2)
    dt = r * dx ** 2 / alpha
    n = 0
    Tmax = np.max(T)
    t = [0]

    while n * dt < tmax:
        if n == 0:
            Tm = T.copy()
        else:
            Tm = Tn.copy()
        Tn = T.copy()

        T[1:-1, 1:-1] = (2 * r * (Tn[1:-1, 2:] - Tm[1:-1, 1:-1] + Tn[1:-1, :-2]) +
                         2 * beta ** 2 * r * (Tn[2:, 1:-1] - Tm[1:-1, 1:-1] + Tn[:-2, 1:-1]) +
                         Tm[1:-1, 1:-1] + 2 * dt * q[1:-1, 1:-1]) / (1 + 2 * r + 2 * beta ** 2 * r)
        n += 1
        Tmax = np.append(Tmax, np.max(T))
        t = np.append(t, n * dt)

    return t, Tmax, n, T


### Define grid and simulation parameters ###
xmin, xmax, ymin, ymax = -1, 1, -1, 1
nx, ny = 41, 41
dx = (xmax - xmin) / nx
beta = 1
x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

alpha = 1
tmax = 3
T0, Q0 = 1, 200

q = Q0 * Q(X, Y)
T = ICBC(X, Y, nx, ny, T0, Q0)

print('Running DuFort-Frankel...')
r = 2
tic = time.perf_counter()
resultDF = DF(T, q, alpha, beta, dx, tmax, r)
toc = time.perf_counter()
print(f'DuFort-Frankel finished in {resultDF[2]} timesteps. Execution took {toc - tic} seconds')

# Plot results
plt.figure(figsize=(4, 3), dpi=200)
plt.loglog(resultDF[0], resultDF[1], ls='dashed', label='DuFort-Frankel')
plt.xlabel('$t$ (s)')
plt.ylabel('$T_{max}$ ($^{\circ}$C)')
plt.legend(frameon=False, fontsize=8)

# Plot only the final temperature distribution
plt.figure(figsize=(5, 4), dpi=200)
c = plt.contourf(X, Y, resultDF[-1], cmap='coolwarm')
plt.colorbar(label='$T$ ($^{\circ}$C)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Temperature Distribution at final time: {tmax} s')
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
output_file = os.path.join(OUTPUT_FOLDER, f"T_{python_filename}.npy")

# Save the array u in the results folder
np.save(output_file, T)

