import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Constants
H = 2.0
n = 100
rho = 1.0
mu = 1.0
C_mu = 0.09
C_e1 = 1.44
C_e2 = 1.92
sigma_k = 1.0
sigma_epsilon = 1.3
f_mu = 1.0
f1 = 1.0
f2 = 1.0
max_iter = 10000
tol = 1e-6

# Non-uniform grid clustered near walls
beta = 1.5
xi = np.linspace(0, 1, n)
y = H * (xi - (1 / (2 * beta)) * np.sin(2 * beta * np.pi * xi))
y = np.sort(y)
dy = np.diff(y)

# Velocity profile and its derivative
u = 1.0 - np.cos(np.pi * y / H)
du_dy = np.zeros_like(y)
du_dy[1:-1] = (u[2:] - u[:-2]) / (y[2:] - y[:-2])
du_dy[0] = (u[1] - u[0]) / dy[0]
du_dy[-1] = (u[-1] - u[-2]) / dy[-1]

# Initialize k and epsilon with small positive values
k = np.full(n, 1e-6)
epsilon = np.full(n, 1e-6)

# Boundary conditions: set small positive values
k[0] = 1e-6
k[-1] = 1e-6
epsilon[0] = 1e-6
epsilon[-1] = 1e-6

for iteration in range(max_iter):
    k_old = k.copy()
    epsilon_old = epsilon.copy()
    
    # Compute mu_t and Pk
    mu_t = C_mu * f_mu * rho * k**2 / (epsilon + 1e-12)
    Pk = mu_t * du_dy**2
    
    # Assemble matrices for k
    A_k = lil_matrix((n, n))
    b_k = rho * epsilon - Pk
    
    # Boundary conditions for k: Dirichlet
    A_k[0,0] = 1.0
    b_k[0] = 1e-6
    A_k[-1,-1] = 1.0
    b_k[-1] = 1e-6

    for i in range(1, n-1):
        dy_w = y[i] - y[i-1]
        dy_e = y[i+1] - y[i]
        a_w = (mu + mu_t[i-1] / sigma_k) / dy_w
        a_e = (mu + mu_t[i] / sigma_k) / dy_e
        
        A_k[i, i-1] = a_w
        A_k[i, i] = -(a_w + a_e)
        A_k[i, i+1] = a_e
        
    # Convert to CSR format
    A_k_csr = A_k.tocsr()
    
    # Solve for k
    try:
        k_new = spsolve(A_k_csr, b_k)
    except:
        k_new = k.copy()
    
    # Ensure positivity
    k_new = np.maximum(k_new, 1e-12)
    
    # Assemble matrices for epsilon
    A_e = lil_matrix((n, n))
    # Compute source term
    source = (epsilon / (k_new + 1e-12)) * (C_e1 * f1 * Pk - C_e2 * f2 * epsilon)
    b_epsilon = -source
    
    # Boundary conditions for epsilon: Dirichlet
    A_e[0,0] = 1.0
    b_epsilon[0] = 1e-6
    A_e[-1,-1] = 1.0
    b_epsilon[-1] = 1e-6
    
    for i in range(1, n-1):
        dy_w = y[i] - y[i-1]
        dy_e = y[i+1] - y[i]
        a_w = (mu + mu_t[i-1] / sigma_epsilon) / dy_w
        a_e = (mu + mu_t[i] / sigma_epsilon) / dy_e
        
        A_e[i, i-1] = a_w
        A_e[i, i] = -(a_w + a_e)
        A_e[i, i+1] = a_e
        
    # Convert to CSR format
    A_e_csr = A_e.tocsr()
    
    # Solve for epsilon
    try:
        epsilon_new = spsolve(A_e_csr, b_epsilon)
    except:
        epsilon_new = epsilon.copy()
    
    # Ensure positivity
    epsilon_new = np.maximum(epsilon_new, 1e-12)
    
    # Update variables
    k = k_new
    epsilon = epsilon_new
    
    # Check convergence
    if np.max(np.abs(k - k_old)) < tol and np.max(np.abs(epsilon - epsilon_old)) < tol:
        break

# Save the final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/k_Fully_Developed_Turbulent_Channel_Flow_KE.npy', k)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/epsilon_Fully_Developed_Turbulent_Channel_Flow_KE.npy', epsilon)