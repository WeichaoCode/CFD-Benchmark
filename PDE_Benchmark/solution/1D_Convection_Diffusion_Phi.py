"""
Solution file for 1d, steady, convection-diffusion solver
"""

import numpy as np
import A02_solver as solver
import matplotlib.pyplot as plt

# Input parameters
L = 1.0  # m, length
rho = 1.0  # kg/m^3, density
gamma = 0.1  # kg/ms, diffusive coefficient
BCa = 1.0  # lHS BC
BCb = 0.0  # RHS BC
u = 2.5  # m/s, velocity
N = 5

# Analytic solution
x = np.linspace(0, L, 100)
phi = BCa + (BCb - BCa) * ((np.exp(rho * u * x / gamma) - 1) / (np.exp(rho * u * L / gamma) - 1))

# Numerical solution
scheme = 1
x_a, phi_a = solver.solve_phi_array(L, N, gamma, rho, u, BCa, BCb, scheme)

# plot results
fig = plt.figure(1, figsize=(6, 4))
plt.plot(x, phi, 'k', label='Exact')
plt.plot(x_a, phi_a, '*r', label='FVM')

plt.xlabel("Length [m]", fontsize=14)
plt.ylabel('\phi', fontsize=14)
plt.legend(fontsize=14, loc=3)
plt.title('Example 2: convection vs diffusion')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
if u < 1:
    plt.text(0.3, 0.95, "u = 0.1 m/s, CDS", fontsize=14, bbox=props)
elif scheme == 1:
    plt.text(0.3, 2.35, "u = 2.5 m/s, CDS", fontsize=14, bbox=props)
else:
    plt.text(0.3, 0.0, "u = 2.5 m/s, UDS", fontsize=14, bbox=props)

import os

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "results")

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# Get the current Python file name without extension
python_filename = os.path.splitext(os.path.basename(__file__))[0]

# Define the file name dynamically
output_file = os.path.join(OUTPUT_FOLDER, f"phi_{python_filename}.npy")

# Save the array u in the results folder
np.save(output_file, phi_a)