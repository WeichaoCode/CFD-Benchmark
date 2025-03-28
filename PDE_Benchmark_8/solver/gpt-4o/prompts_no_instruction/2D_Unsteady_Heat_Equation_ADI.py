import numpy as np

# Parameters
alpha = 0.01  # Thermal diffusivity
Q0 = 200.0  # Source term coefficient
sigma = 0.1  # Source term spread
nx, ny = 41, 41  # Grid resolution
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_max = 3.0  # Maximum time
r = 0.25  # Stability parameter

# Derived parameters
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
dt = r * dx**2 / alpha
nt = int(t_max / dt) + 1

# Create grid
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
T = np.zeros((ny, nx))

# Source term
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# ADI method
for n in range(nt):
    # Step 1: Implicit in x, explicit in y
    T_half = np.zeros_like(T)
    for j in range(1, ny-1):
        A = np.zeros((nx, nx))
        b = np.zeros(nx)
        for i in range(1, nx-1):
            A[i, i-1] = -r / 2
            A[i, i] = 1 + r
            A[i, i+1] = -r / 2
            b[i] = (1 - r) * T[j, i] + (r / 2) * (T[j+1, i] + T[j-1, i]) + dt * q[j, i]
        A[0, 0] = A[-1, -1] = 1.0
        b[0] = b[-1] = 0.0
        T_half[j, :] = np.linalg.solve(A, b)

    # Step 2: Implicit in y, explicit in x
    for i in range(1, nx-1):
        A = np.zeros((ny, ny))
        b = np.zeros(ny)
        for j in range(1, ny-1):
            A[j, j-1] = -r / 2
            A[j, j] = 1 + r
            A[j, j+1] = -r / 2
            b[j] = (1 - r) * T_half[j, i] + (r / 2) * (T_half[j, i+1] + T_half[j, i-1]) + dt * q[j, i]
        A[0, 0] = A[-1, -1] = 1.0
        b[0] = b[-1] = 0.0
        T[:, i] = np.linalg.solve(A, b)

# Save the final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/T_2D_Unsteady_Heat_Equation_ADI.npy', T)