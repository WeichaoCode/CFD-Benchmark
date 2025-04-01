import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Parameters
H = 2.0
n = 100
C_e1 = 1.44
C_e2 = 1.92
C_mu = 0.09
sigma_k = 1.0
sigma_epsilon = 1.3
C1 = 1.0
C2 = 1.0
rho = 1.0
mu = 0.01
L = 0.5
T_val = 1.0
v_t = 1.0  # Assuming turbulent temperature

# Mesh generation with clustering near the walls
beta = 3.0
i = np.arange(n)
y = H * (i / (n - 1))**beta
dy = np.diff(y)
dy = np.append(dy, dy[-1])

# Initialize variables with small positive values to avoid division by zero
k = np.ones(n) * 1e-6
epsilon = np.ones(n) * 1e-6
v2 = np.ones(n) * 1e-6
f = np.ones(n) * 1e-6

# Boundary conditions for velocity (u=0 at walls)
u_left = 0.0
u_right = 0.0

# Iterative solver parameters
max_iter = 1000
tolerance = 1e-6

for iteration in range(max_iter):
    k_old = k.copy()
    epsilon_old = epsilon.copy()
    v2_old = v2.copy()
    f_old = f.copy()
    
    # Compute mu_t
    mu_t_k = C_mu * rho * np.sqrt(epsilon / (k + 1e-12)) * v_t
    mu_t_epsilon = C_mu * rho * np.sqrt(epsilon / (k + 1e-12)) * v_t
    
    # Assemble the linear system
    size = 4 * n
    A = lil_matrix((size, size))
    b = np.zeros(size)
    
    for j in range(n):
        # Indices for variables
        idx_k = j
        idx_epsilon = n + j
        idx_v2 = 2 * n + j
        idx_f = 3 * n + j
        
        if j == 0 or j == n-1:
            # Boundary conditions
            A[idx_k, idx_k] = 1
            b[idx_k] = 0
            A[idx_epsilon, idx_epsilon] = 1
            b[idx_epsilon] = 0
            A[idx_v2, idx_v2] = 1
            b[idx_v2] = 0
            A[idx_f, idx_f] = 1
            b[idx_f] = 0
        else:
            # Coefficients for k equation
            A[idx_k, j-1] = (mu + mu_t_k[j]) / dy[j-1]**2
            A[idx_k, j] = -2 * (mu + mu_t_k[j]) * (1/dy[j-1]**2 + 1/dy[j]**2)
            A[idx_k, j+1] = (mu + mu_t_k[j]) / dy[j]**2
            A[idx_k, n + j] = -rho
            b[idx_k] = 0
            
            # Coefficients for epsilon equation
            A[idx_epsilon, n + j -1] = (mu + mu_t_epsilon[j]) / dy[j-1]**2
            A[idx_epsilon, n + j] = -2 * (mu + mu_t_epsilon[j]) * (1/dy[j-1]**2 + 1/dy[j]**2) + C_e2 * rho / T_val
            A[idx_epsilon, n + j +1] = (mu + mu_t_epsilon[j]) / dy[j]**2
            A[idx_epsilon, j] = C_e1 / T_val * mu_t_k[j]
            b[idx_epsilon] = 0
            
            # Coefficients for v2 equation
            A[idx_v2, 2*n + j -1] = (mu + mu_t_k[j]) / dy[j-1]**2
            A[idx_v2, 2*n + j] = -2 * (mu + mu_t_k[j]) * (1/dy[j-1]**2 + 1/dy[j]**2) + 6 * rho * epsilon[j] / (k[j] + 1e-12)
            A[idx_v2, 2*n + j +1] = (mu + mu_t_k[j]) / dy[j]**2
            A[idx_v2, j] = rho * f[j]
            b[idx_v2] = 0
            
            # Coefficients for f equation
            A[idx_f, 3*n + j -1] = L**2 / dy[j-1]**2
            A[idx_f, 3*n + j] = -2 * L**2 / dy[j-1]**2 -1
            A[idx_f, 3*n + j +1] = L**2 / dy[j]**2
            A[idx_f, n + j] = -C2
            A[idx_f, 2*n + j] = -C1 / T_val
            b[idx_f] = (C1 * (6 - v2[j]) - (2/3)*(C1 -1)) / T_val
    
    # Convert A to CSR format for efficient solving
    A_csr = A.tocsr()
    
    # Solve the linear system
    try:
        u = spsolve(A_csr, b)
    except:
        break
    
    # Update variables
    k = u[0:n]
    epsilon = u[n:2*n]
    v2 = u[2*n:3*n]
    f = u[3*n:4*n]
    
    # Check for convergence
    if (np.linalg.norm(k - k_old, np.inf) < tolerance and
        np.linalg.norm(epsilon - epsilon_old, np.inf) < tolerance and
        np.linalg.norm(v2 - v2_old, np.inf) < tolerance and
        np.linalg.norm(f - f_old, np.inf) < tolerance):
        break

# Compute mu_t with final values
mu_t = C_mu * rho * np.sqrt(epsilon / (k + 1e-12)) * v_t

# Compute velocity profile u(y) assuming fully developed flow
# Integration of du/dy = mu_total^{-1} * dp/dx
# Assuming pressure gradient dp/dx is constant, set dp/dx = 1 for simplicity
dp_dx = 1.0
mu_total = mu + mu_t / sigma_k
u = np.zeros(n)
for j in range(1, n):
    u[j] = u[j-1] + (dp_dx / mu_total[j]) * dy[j-1]

# Save the final solutions
save_values = ['k', 'epsilon', 'v2', 'f', 'u']
np.save('k.npy', k)
np.save('epsilon.npy', epsilon)
np.save('v2.npy', v2)
np.save('f.npy', f)
np.save('u.npy', u)