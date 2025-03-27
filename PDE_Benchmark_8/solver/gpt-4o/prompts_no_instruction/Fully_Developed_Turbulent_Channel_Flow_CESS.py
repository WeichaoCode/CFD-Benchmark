import numpy as np

# Parameters
H = 2.0  # Height of the domain
n = 100  # Number of grid points
mu = 1.0e-3  # Molecular viscosity (example value)

# Create a non-uniform grid clustered near the walls
y = np.linspace(0, H, n)
y = H * (np.sinh(3 * (y / H - 0.5)) / np.sinh(1.5) + 0.5)

# Initialize velocity field
u = np.zeros(n)

# Cess model parameters (example values)
kappa = 0.41
A = 26.0

# Function to compute turbulent viscosity using Cess model
def compute_mu_t(y, H, mu):
    y_plus = y * (1 / mu)
    return mu * (1 + (A * y_plus * (1 - y / H))**2)

# Finite Difference Method
def solve_pde(y, u, mu, H, n, tol=1e-6, max_iter=10000):
    dy = np.diff(y)
    dy_mid = (dy[:-1] + dy[1:]) / 2.0
    u_new = np.copy(u)
    
    for iteration in range(max_iter):
        mu_t = compute_mu_t(y, H, mu)
        mu_eff = mu + mu_t
        
        # Construct the tridiagonal matrix
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        for i in range(1, n-1):
            A[i, i-1] = mu_eff[i-1] / dy[i-1]
            A[i, i] = -(mu_eff[i-1] / dy[i-1] + mu_eff[i] / dy[i])
            A[i, i+1] = mu_eff[i] / dy[i]
            b[i] = -1
        
        # Apply boundary conditions
        A[0, 0] = A[-1, -1] = 1.0
        b[0] = b[-1] = 0.0
        
        # Solve the linear system
        u_new = np.linalg.solve(A, b)
        
        # Check for convergence
        if np.linalg.norm(u_new - u, ord=np.inf) < tol:
            break
        
        u = u_new
    
    return u_new

# Solve the PDE
u_final = solve_pde(y, u, mu, H, n)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_final_Fully_Developed_Turbulent_Channel_Flow_CESS.npy', u_final)