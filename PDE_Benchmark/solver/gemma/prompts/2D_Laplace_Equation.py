import numpy as np
from numpy import array
import matplotlib.pyplot as plt

# Define domain and grid
nx = 50
ny = 50
x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Initialize solution
p = np.zeros((ny, nx))

# Set boundary conditions
p[:, 0] = 0
p[:, -1] = Y

# Finite difference discretization
dx = x[1] - x[0]
dy = y[1] - y[0]
laplacian = (1 / (dx**2) + 1 / (dy**2))

# Solve using iterative method (e.g., Gauss-Seidel)
tolerance = 1e-6
max_iterations = 1000
for iteration in range(max_iterations):
    p_new = p.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            p_new[j, i] = (p_new[j, i - 1] + p_new[j, i + 1] + p_new[j - 1, i] + p_new[j + 1, i]) * (1 / (4 * laplacian))
    error = np.max(np.abs(p_new - p))
    if error < tolerance:
        break
    p = p_new

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/p_2D_Laplace_Equation.npy', p)