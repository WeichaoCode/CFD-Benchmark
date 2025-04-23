import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
Re_tau = 395
mu = 1 / Re_tau
kappa = 0.42
A = 25.4
y_max = 2.0
n_points = 100
dy = y_max / (n_points - 1)
y = np.linspace(0, y_max, n_points)

# Initial conditions
u = np.zeros(n_points)
mu_t = np.zeros(n_points)

# Function to calculate mu_eff
def mu_eff(y, Re_tau, mu):
    y_plus = y * Re_tau
    term1 = (1 + (1/9) * kappa**2 * Re_tau**2 * (2*y - y**2)**2 * (3 - 4*y + 2*y**2)**2 * (1 - np.exp(-y_plus/A))**2)
    return mu * (0.5 * np.sqrt(term1) - 0.5)

# Iterative solver
tolerance = 1e-6
max_iterations = 1000
for iteration in range(max_iterations):
    mu_eff_values = mu_eff(y, Re_tau, mu)
    
    # Construct the finite difference matrix
    diag_main = np.zeros(n_points)
    diag_lower = np.zeros(n_points - 1)
    diag_upper = np.zeros(n_points - 1)
    
    for i in range(1, n_points - 1):
        mu_eff_mid = (mu_eff_values[i] + mu_eff_values[i+1]) / 2
        mu_eff_prev = (mu_eff_values[i] + mu_eff_values[i-1]) / 2
        
        diag_main[i] = -mu_eff_mid / dy**2 - mu_eff_prev / dy**2
        diag_lower[i-1] = mu_eff_prev / dy**2
        diag_upper[i] = mu_eff_mid / dy**2
    
    # Boundary conditions
    diag_main[0] = 1.0
    diag_main[-1] = 1.0
    
    # Right-hand side
    rhs = np.full(n_points, -1.0)
    rhs[0] = 0.0
    rhs[-1] = 0.0
    
    # Solve the linear system
    A_matrix = diags([diag_lower, diag_main, diag_upper], offsets=[-1, 0, 1], format='csc')
    u_new = spsolve(A_matrix, rhs)
    
    # Check for convergence
    if np.linalg.norm(u_new - u, ord=np.inf) < tolerance:
        u = u_new
        break
    
    u = u_new

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)