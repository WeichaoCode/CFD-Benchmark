import numpy as np
from scipy.sparse import diags, lil_matrix
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
    term1 = 1 + (1/9) * kappa**2 * Re_tau**2 * (2*y - y**2)**2 * (3 - 4*y + 2*y**2)**2
    term2 = (1 - np.exp(-y_plus(y)/A))**2
    return mu * (0.5 * (term1 * term2)**0.5 - 0.5)

# Discretize the PDE using finite difference method
def solve_momentum_equation():
    # Coefficients for the finite difference scheme
    mu_eff_values = mu_eff(y)
    dmu_eff_dy = np.gradient(mu_eff_values, dy)
    
    # Construct the sparse matrix using LIL format for efficiency
    A_matrix = lil_matrix((num_points, num_points))
    
    # Fill the matrix
    for i in range(1, num_points - 1):
        A_matrix[i, i-1] = mu_eff_values[i] / dy**2 - dmu_eff_dy[i] / (2*dy)
        A_matrix[i, i] = -2 * mu_eff_values[i] / dy**2
        A_matrix[i, i+1] = mu_eff_values[i] / dy**2 + dmu_eff_dy[i] / (2*dy)
    
    # Right-hand side
    b = -np.ones(num_points)
    
    # Apply boundary conditions
    A_matrix[0, 0] = 1
    b[0] = 0
    
    A_matrix[-1, -1] = 1
    b[-1] = 0
    
    # Convert to CSC format for solving
    A_matrix = A_matrix.tocsc()
    
    # Solve the linear system
    u[:] = spsolve(A_matrix, b)

# Solve the problem
solve_momentum_equation()

# Save the solution
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)