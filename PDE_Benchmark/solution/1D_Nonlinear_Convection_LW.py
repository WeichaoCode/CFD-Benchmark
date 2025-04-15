"""
1D non-linear convection equation
---------------------------------------------------------------
- Exercise 5 - Finite difference solution of the one-dimensional non-linear convection equation
- Lax-Wendroff method
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""
import numpy as np
import matplotlib.pyplot as plt
import math

nu = 0.5       # CFL number
dt = 0.01      # time step
T = 500        # max number of time steps
dx = dt/nu     # space step
L = 2*math.pi  # domain length

## Define space vector
x = np.linspace(0,L,math.ceil(L/(dt/nu)))
dx = x[1]-x[0]
print('Step size =', dx, 's')
nx = x.size
print('Number of x nodes =', nx)

## Define initial conditions
u0 = np.sin(x) + 0.5*np.sin(0.5*x)
u = np.copy(u0)
ustar = np.zeros(u.size)
unp1 = np.zeros(u.size)
for t in range(0, T):
    Ajp = 0.5 * (u[0] + u[1])
    Ajm = 0.5 * (u[0] + u[nx - 2])
    unp1[0] = u[0] - 0.25 * nu * (u[1] ** 2 - u[nx - 2] ** 2) \
              + 0.5 * (nu ** 2) * (0.5 * Ajp * (u[1] ** 2 - u[0] ** 2) \
                                   - 0.5 * Ajm * (u[0] ** 2 - u[nx - 2] ** 2)
                                   )
    for i in range(1, nx - 1):
        Ajp = 0.5 * (u[i] + u[i + 1])
        Ajm = 0.5 * (u[i] + u[i - 1])
        unp1[i] = u[i] - 0.25 * nu * (u[i + 1] ** 2 - u[i - 1] ** 2) \
                  + 0.5 * (nu ** 2) * (0.5 * Ajp * (u[i + 1] ** 2 - u[i] ** 2) \
                                       - 0.5 * Ajm * (u[i] ** 2 - u[i - 1] ** 2)
                                       )
    Ajp = 0.5 * (u[i] + u[1])
    Ajm = 0.5 * (u[i] + u[i - 1])
    unp1[i] = u[i] - 0.25 * nu * (u[1] ** 2 - u[i - 1] ** 2) \
              + 0.5 * (nu ** 2) * (0.5 * Ajp * (u[1] ** 2 - u[i] ** 2) \
                                   - 0.5 * Ajm * (u[i] ** 2 - u[i - 1] ** 2)
                                   )
    u = np.copy(unp1)

    if (t % 100 == 0):
        string = '$t =$' + str(t * dt) + ' s'
        plt.plot(x, u, label=string)

uLW = np.copy(u)
string = '$t =$' + str((t + 1) * dt) + ' s'
plt.plot(x, u, label=string)
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

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