import numpy as np

# Parameters
mu = 1e-3  # dynamic viscosity (Pa·s)
dPdz = -3.2  # pressure gradient (Pa/m)
h = 0.1  # domain height (m)
n_x = n_y = 80  # number of grid points
dx = dy = h / (n_x - 1)  # grid spacing

# Initialize the velocity field
w = np.zeros((n_x, n_y))

# Finite Volume Method (FVM) setup
tolerance = 1e-6
max_iterations = 10000
omega = 1.5  # relaxation factor for SOR

# Coefficients for the FVM discretization
aP = np.zeros((n_x, n_y))
aE = np.zeros((n_x, n_y))
aW = np.zeros((n_x, n_y))
aN = np.zeros((n_x, n_y))
aS = np.zeros((n_x, n_y))
b = np.zeros((n_x, n_y))

# Fill the coefficients
for i in range(1, n_x - 1):
    for j in range(1, n_y - 1):
        aE[i, j] = mu / dx**2
        aW[i, j] = mu / dx**2
        aN[i, j] = mu / dy**2
        aS[i, j] = mu / dy**2
        aP[i, j] = aE[i, j] + aW[i, j] + aN[i, j] + aS[i, j]
        b[i, j] = -dPdz

# Iterative solver (Successive Over-Relaxation)
for iteration in range(max_iterations):
    w_old = w.copy()
    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            w_new = (aE[i, j] * w[i + 1, j] +
                     aW[i, j] * w[i - 1, j] +
                     aN[i, j] * w[i, j + 1] +
                     aS[i, j] * w[i, j - 1] +
                     b[i, j]) / aP[i, j]
            w[i, j] = (1 - omega) * w[i, j] + omega * w_new
    
    # Check for convergence
    if np.linalg.norm(w - w_old, ord=np.inf) < tolerance:
        print(f"Converged after {iteration} iterations.")
        break
else:
    print("Did not converge within the maximum number of iterations.")

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/w_2D_Diffusion_FVM.npy', w)