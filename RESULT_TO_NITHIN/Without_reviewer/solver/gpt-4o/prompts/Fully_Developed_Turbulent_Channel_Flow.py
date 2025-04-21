import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
Re_tau = 395
mu = 1 / Re_tau
kappa = 0.42
A = 25.4
y_plus = lambda y: y * Re_tau

# Domain
y_start, y_end = 0, 2
num_points = 100
y = np.linspace(y_start, y_end, num_points)
dy = y[1] - y[0]

# Initial conditions
u = np.zeros(num_points)
mu_t = np.zeros(num_points)

# Cess turbulence model
def mu_eff(y):
    y_plus_val = y_plus(y)
    term1 = (1 + (1/9) * kappa**2 * Re_tau**2 * (2*y - y**2)**2 * (3 - 4*y + 2*y**2)**2 * (1 - np.exp(-y_plus_val/A))**2)
    return mu * (0.5 * np.sqrt(term1) - 0.5)

# Discretize the PDE using finite difference method
def build_matrix_and_rhs():
    mu_eff_values = mu_eff(y)
    main_diag = np.zeros(num_points)
    upper_diag = np.zeros(num_points - 1)
    lower_diag = np.zeros(num_points - 1)
    rhs = np.full(num_points, -1.0)

    for i in range(1, num_points - 1):
        dmu_dy = (mu_eff_values[i+1] - mu_eff_values[i-1]) / (2 * dy)
        main_diag[i] = -2 * mu_eff_values[i] / dy**2
        upper_diag[i-1] = mu_eff_values[i] / dy**2 + dmu_dy / (2 * dy)
        lower_diag[i-1] = mu_eff_values[i] / dy**2 - dmu_dy / (2 * dy)

    # Apply boundary conditions
    main_diag[0] = main_diag[-1] = 1.0
    rhs[0] = rhs[-1] = 0.0

    # Construct sparse matrix
    diagonals = [main_diag, upper_diag, lower_diag]
    A = diags(diagonals, [0, 1, -1], format='csc')
    return A, rhs

# Solve the linear system
A, rhs = build_matrix_and_rhs()
u = spsolve(A, rhs)

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)