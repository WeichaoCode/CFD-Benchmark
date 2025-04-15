import numpy as np

# Parameters
nx, ny = 80, 80
h = 0.1
dx = h / (nx - 1)
dy = dx
mu = 1e-3
dPdz = -3.2
S = dPdz / mu
tol = 1e-6
max_iterations = 10000

# Initialize
w = np.zeros((ny, nx))
w_new = np.zeros_like(w)

# Iterative solver
for _ in range(max_iterations):
    w_new[1:-1,1:-1] = 0.25 * (w[2:,1:-1] + w[:-2,1:-1] + w[1:-1,2:] + w[1:-1,:-2] - (dx**2) * S)
    # Apply boundary conditions (Dirichlet w=0)
    w_new[0, :] = 0
    w_new[-1, :] = 0
    w_new[:, 0] = 0
    w_new[:, -1] = 0
    # Check convergence
    if np.max(np.abs(w_new - w)) < tol:
        break
    w, w_new = w_new, w

# Save the final solution
save_values = ['w']
for var in save_values:
    np.save(f"{var}.npy", w)