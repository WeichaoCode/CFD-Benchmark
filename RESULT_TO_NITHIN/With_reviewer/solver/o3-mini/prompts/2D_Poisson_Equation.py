#!/usr/bin/env python3
import numpy as np

# Domain parameters
Lx = 2.0
Ly = 1.0

# Grid resolution
nx = 101  # number of points in x-direction
ny = 51   # number of points in y-direction
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Create grids
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize pressure field and source term b (2D arrays)
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Set the source term at specified points
# Note: the array index ordering is [j, i] where j corresponds to y and i corresponds to x.
# Find the indices that are closest to the specified coordinates.
i1 = int(round((Lx/4) / dx))
j1 = int(round((Ly/4) / dy))
i2 = int(round((3*Lx/4) / dx))
j2 = int(round((3*Ly/4) / dy))

b[j1, i1] = 100.0
b[j2, i2] = -100.0

# Iterative solver parameters
tol = 1e-4
max_iter = 10000
error = 1.0
iter_count = 0

# Finite difference coefficient
dx2 = dx * dx
dy2 = dy * dy
coef = 1.0 / (2*(dx2 + dy2))

# Solve using Gauss-Seidel iteration method
while error > tol and iter_count < max_iter:
    error = 0.0
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            p_old = p[j, i]
            p[j, i] = ((dy2 * (p[j, i+1] + p[j, i-1]) +
                        dx2 * (p[j+1, i] + p[j-1, i]) -
                        dx2 * dy2 * b[j, i]) * coef)
            error = max(error, abs(p[j, i] - p_old))
    iter_count += 1

# Save the resulting 2D pressure field "p" to an .npy file.
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Poisson_Equation.npy', p)