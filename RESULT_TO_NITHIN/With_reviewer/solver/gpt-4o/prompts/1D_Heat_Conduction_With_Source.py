import numpy as np

# Parameters
k = 1000  # Thermal conductivity in W/(m*K)
Q = 2e6   # Heat generation in W/m^3
L = 0.5   # Length of the rod in meters
T0 = 100  # Temperature at x=0 in degrees Celsius
TL = 200  # Temperature at x=L in degrees Celsius

# Discretization
N = 100  # Number of control volumes
dx = L / N  # Width of each control volume

# Coefficients for the finite volume method
A = k / dx
B = Q * dx

# Initialize temperature array
T = np.zeros(N + 1)

# Apply boundary conditions
T[0] = T0
T[-1] = TL

# Construct the coefficient matrix and right-hand side vector
A_matrix = np.zeros((N + 1, N + 1))
b_vector = np.zeros(N + 1)

# Fill the matrix and vector
for i in range(1, N):
    A_matrix[i, i - 1] = A
    A_matrix[i, i] = -2 * A
    A_matrix[i, i + 1] = A
    b_vector[i] = -B

# Apply boundary conditions to the matrix
A_matrix[0, 0] = 1
A_matrix[-1, -1] = 1
b_vector[0] = T0
b_vector[-1] = TL

# Solve the linear system
T = np.linalg.solve(A_matrix, b_vector)

# Save the temperature distribution
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/T_1D_Heat_Conduction_With_Source.npy', T)