#!/usr/bin/env python3
import numpy as np

# Given parameters
k = 1000.0            # thermal conductivity, W/(m·K)
h = 62.5              # convective heat transfer coefficient, W/(m^2·K)
T_inf = 20.0          # ambient temperature, °C
# Given convective term coefficient: hP/(kA)=25.0 => hP/A = 25.0 * k
conv_coef = 25.0 * k  # hP/A, effective convective sink coefficient, W/m^3 (in this formulation)

# Domain and boundary conditions
L = 0.5               # Length of the rod [m]
T0 = 100.0            # Temperature at x=0, °C
T_L = 200.0           # Temperature at x=L, °C

# Number of nodes
N = 51                # number of grid points
dx = L / (N - 1)      # uniform grid spacing

# Create array for x positions
x = np.linspace(0, L, N)

# Coefficients for finite volume discretization (central scheme)
a_coeff = k / dx**2                     # coefficient for T[i-1] and T[i+1]
b_coeff = -2.0 * k / dx**2 - conv_coef    # coefficient for T[i]
source = conv_coef * T_inf               # source term from convection

# Assemble the linear system for the interior nodes
N_int = N - 2  # number of internal nodes (excluding boundaries)
A = np.zeros((N_int, N_int))
b = np.full(N_int, source)

for i in range(N_int):
    # Main diagonal
    A[i, i] = b_coeff
    # Lower diagonal
    if i > 0:
        A[i, i-1] = a_coeff
    # Upper diagonal
    if i < N_int - 1:
        A[i, i+1] = a_coeff

# Adjust right-hand side for known boundary conditions
# For the first internal node (corresponding to x[1])
b[0] -= a_coeff * T0
# For the last internal node (corresponding to x[N-2])
b[-1] -= a_coeff * T_L

# Solve the linear system for interior temperatures
T_internal = np.linalg.solve(A, b)

# Construct the full temperature solution including boundary values
T = np.zeros(N)
T[0] = T0
T[1:-1] = T_internal
T[-1] = T_L

# Save the final solution in a .npy file (1D array)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/T_1D_Heat_Conduction_With_Convection.npy', T)

# End of code