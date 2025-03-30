import numpy as np
from numpy.linalg import solve

# Constants
rho = 1.0
mu = 0.01
sigma_k = 1.0
sigma_epsilon = 1.3
C_e1 = 1.44
C_e2 = 1.92
T_const = 1.0
C_mu = 0.09
C1 = 1.0
C2 = 1.0
L = 1.0
H = 2.0
n = 100
tolerance = 1e-6
max_iterations = 10000

# Create non-uniform mesh clustered near walls
beta = 1.5
y_linear = np.linspace(0, 1, n)
y = H * (y_linear ** beta)
dy = np.diff(y)
dy = np.append(dy, dy[-1])  # To make dy the same size as y

# Initialize variables
k = np.zeros(n)
epsilon = np.zeros(n)
v2 = np.zeros(n)
f = np.zeros(n)
mu_t = np.zeros(n)
Pk = np.zeros(n)

# Function to compute mu_t
def compute_mu_t(rho, epsilon, k, T_t):
    with np.errstate(divide='ignore', invalid='ignore'):
        mu_t = C_mu * rho * np.sqrt(epsilon / k) * T_t
        mu_t[np.isnan(mu_t)] = 0.0
        mu_t[np.isinf(mu_t)] = 0.0
    return mu_t

# Iterative solver
for iteration in range(max_iterations):
    k_old = k.copy()
    epsilon_old = epsilon.copy()
    v2_old = v2.copy()
    f_old = f.copy()
    
    # Compute mu_t
    mu_t = compute_mu_t(rho, epsilon, k, T_const)
    
    # Compute Pk
    Pk = rho * epsilon
    
    # Update k
    A_k = np.zeros((n, n))
    b_k = Pk.copy()
    for i in range(1, n-1):
        A_k[i, i-1] = (mu + mu_t[i]/sigma_k) / dy[i-1]**2
        A_k[i, i] = -2 * (mu + mu_t[i]/sigma_k) * (1/dy[i-1]**2 + 1/dy[i]**2)
        A_k[i, i+1] = (mu + mu_t[i]/sigma_k) / dy[i]**2
    A_k[0,0] = 1
    A_k[-1,-1] = 1
    b_k[0] = 0
    b_k[-1] = 0
    k = solve(A_k, b_k)
    
    # Update epsilon
    A_e = np.zeros((n, n))
    b_e = (C_e1 * Pk - C_e2 * rho * epsilon) / T_const
    for i in range(1, n-1):
        A_e[i, i-1] = (mu + mu_t[i]/sigma_epsilon) / dy[i-1]**2
        A_e[i, i] = -2 * (mu + mu_t[i]/sigma_epsilon) * (1/dy[i-1]**2 + 1/dy[i]**2) + (C_e2 * rho) / T_const
        A_e[i, i+1] = (mu + mu_t[i]/sigma_epsilon) / dy[i]**2
    A_e[0,0] = 1
    A_e[-1,-1] = 1
    b_e[0] = 0
    b_e[-1] = 0
    epsilon = solve(A_e, b_e)
    
    # Update v2
    A_v2 = np.zeros((n, n))
    b_v2 = np.zeros(n)
    mask = k > 1e-8
    b_v2[mask] = 6 * rho * v2[mask] * epsilon[mask] / k[mask]
    for i in range(1, n-1):
        A_v2[i, i-1] = (mu + mu_t[i]/sigma_k) / dy[i-1]**2
        A_v2[i, i] = -2 * (mu + mu_t[i]/sigma_k) * (1/dy[i-1]**2 + 1/dy[i]**2)
        A_v2[i, i+1] = (mu + mu_t[i]/sigma_k) / dy[i]**2
    A_v2[0,0] = 1
    A_v2[-1,-1] = 1
    b_v2[0] = 0
    b_v2[-1] = 0
    v2 = solve(A_v2, b_v2)
    
    # Update f
    A_f = np.zeros((n, n))
    b_f = (1/T_const) * (C1 * (6 - v2)) - (2/3) * (C1 -1) - C2 * Pk
    for i in range(1, n-1):
        A_f[i, i-1] = L**2 / dy[i-1]**2
        A_f[i, i] = -2 * L**2 / dy[i-1]**2 - 1
        A_f[i, i+1] = L**2 / dy[i]**2
    A_f[0,0] = 1
    A_f[-1,-1] = 1
    b_f[0] = 0
    b_f[-1] = 0
    f = solve(A_f, b_f)
    
    # Check convergence
    if (np.all(np.abs(k - k_old) < tolerance) and
        np.all(np.abs(epsilon - epsilon_old) < tolerance) and
        np.all(np.abs(v2 - v2_old) < tolerance) and
        np.all(np.abs(f - f_old) < tolerance)):
        print(f'Converged in {iteration+1} iterations.')
        break
else:
    print('Warning: Maximum iterations reached without convergence.')

# Save the final solutions
np.save('k.npy', k)
np.save('epsilon.npy', epsilon)
np.save('v2.npy', v2)
np.save('f.npy', f)
np.save('Pk.npy', Pk)
np.save('mu_t.npy', mu_t)