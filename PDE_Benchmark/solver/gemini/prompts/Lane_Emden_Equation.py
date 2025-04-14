import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
n = 3.0  # Polytropic index
r_max = 1.0  # Outer radius
num_points = 100  # Number of radial points
r = np.linspace(0, r_max, num_points)
dr = r[1] - r[0]

# Initial guess
R0 = 5
f = R0**(2/(n-1)) * (1 - r**2)**2

# Boundary conditions
f[0] = f[1] # Regularity condition at r=0, df/dr = 0
f[-1] = 0.0  # Dirichlet boundary condition at r=1

# Iterative solver (Newton-Raphson)
tolerance = 1e-6
max_iterations = 100
error = 1.0
iteration = 0

while error > tolerance and iteration < max_iterations:
    f_old = np.copy(f)

    # Construct the Jacobian matrix (sparse tridiagonal)
    diag = -2 / dr**2 + n * f**(n-1)
    off_diag = (1 / dr**2) + (1 / (r[1:-1] * dr))
    
    main_diag = diag[1:-1] - 2/(r[1]*dr)* (r[1]==0)
    upper_diag = off_diag[0:-1]
    lower_diag = off_diag[0:-1]

    if r[1] == 0:
        upper_diag[0] = 0
        lower_diag[0] = 0
    
    diagonals = [main_diag, upper_diag, np.concatenate((np.array([0]),lower_diag))]
    offsets = [0, 1, -1]
    J = diags(diagonals, offsets, shape=(num_points-2, num_points-2)).tocsc()

    # Construct the residual vector
    residual = (f[2:] - 2*f[1:-1] + f[:-2]) / dr**2 + (2/r[1:-1]) * (f[2:] - f[:-2]) / (2*dr) + f[1:-1]**n
    
    # Solve the linear system J * delta_f = -residual
    delta_f = spsolve(J, -residual)

    # Update the solution
    f[1:-1] = f[1:-1] + delta_f

    # Apply boundary conditions
    f[0] = f[1] # Regularity condition at r=0
    f[-1] = 0.0  # Dirichlet boundary condition at r=1

    # Calculate the error
    error = np.max(np.abs(f - f_old))
    iteration += 1

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/f_Lane_Emden_Equation.npy', f)