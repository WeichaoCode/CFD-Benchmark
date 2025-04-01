import numpy as np
from scipy.linalg import solve_banded

# Parameters
alpha = 0.01
Q0 = 200.0
sigma = 0.1
nx, ny = 41, 41
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_max = 3.0
r = 0.25

# Discretization
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
dt = r * dx**2 / alpha
nt = int(t_max / dt) + 1

# Grid
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
T = 1 + 200 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Source term
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Boundary conditions
T[:, 0] = 1
T[:, -1] = 1
T[0, :] = 1
T[-1, :] = 1

# ADI method coefficients
rx = alpha * dt / (2 * dx**2)
ry = alpha * dt / (2 * dy**2)

# Time-stepping loop
for n in range(nt):
    # Step 1: Implicit in x, explicit in y
    T_star = np.copy(T)
    for j in range(1, ny-1):
        # Construct the tridiagonal matrix
        A = np.zeros((3, nx-2))
        A[0, 1:] = -rx
        A[1, :] = 1 + 2 * rx
        A[2, :-1] = -rx
        
        # Right-hand side
        b = T[1:-1, j] + ry * (T[1:-1, j+1] - 2 * T[1:-1, j] + T[1:-1, j-1]) + dt * q[1:-1, j]
        b[0] += rx * T[0, j]
        b[-1] += rx * T[-1, j]
        
        # Solve the tridiagonal system
        T_star[1:-1, j] = solve_banded((1, 1), A, b)
    
    # Step 2: Implicit in y, explicit in x
    for i in range(1, nx-1):
        # Construct the tridiagonal matrix
        A = np.zeros((3, ny-2))
        A[0, 1:] = -ry
        A[1, :] = 1 + 2 * ry
        A[2, :-1] = -ry
        
        # Right-hand side
        b = T_star[i, 1:-1] + rx * (T_star[i+1, 1:-1] - 2 * T_star[i, 1:-1] + T_star[i-1, 1:-1]) + dt * q[i, 1:-1]
        b[0] += ry * T_star[i, 0]
        b[-1] += ry * T_star[i, -1]
        
        # Solve the tridiagonal system
        T[i, 1:-1] = solve_banded((1, 1), A, b)

# Save the final temperature field
save_values = ['T']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/T_2D_Unsteady_Heat_Equation_ADI.npy', T)