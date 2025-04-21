import numpy as np

# Parameters
k = 1000.0  # W/(m*K)
hP_kA = 25.0  # m^-2
T_inf = 20.0  # °C
L = 0.5  # m
T0 = 100.0  # °C
T1 = 200.0  # °C

# Discretization
N = 100  # number of control volumes
dx = L / N

# Coefficients
aW = k / dx
aE = k / dx
aP = aW + aE + hP_kA * dx

# Initialize temperature array
T = np.zeros(N + 2)  # Including ghost cells for boundary conditions
T[0] = T0  # Boundary condition at x=0
T[-1] = T1  # Boundary condition at x=L

# Coefficient matrix and right-hand side vector
A = np.zeros((N, N))
b = np.zeros(N)

# Fill the matrix and RHS vector
for i in range(1, N + 1):
    if i == 1:
        A[i - 1, i - 1] = aP
        A[i - 1, i] = -aE
        b[i - 1] = hP_kA * dx * T_inf + aW * T0
    elif i == N:
        A[i - 1, i - 2] = -aW
        A[i - 1, i - 1] = aP
        b[i - 1] = hP_kA * dx * T_inf + aE * T1
    else:
        A[i - 1, i - 2] = -aW
        A[i - 1, i - 1] = aP
        A[i - 1, i] = -aE
        b[i - 1] = hP_kA * dx * T_inf

# Solve the linear system
T_internal = np.linalg.solve(A, b)

# Store the solution in the full temperature array
T[1:-1] = T_internal

# Save the temperature distribution
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/T_1D_Heat_Conduction_With_Convection.npy', T)