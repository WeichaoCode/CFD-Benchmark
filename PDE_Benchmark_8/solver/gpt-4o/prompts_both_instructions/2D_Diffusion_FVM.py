import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 1e-3  # dynamic viscosity (Pa·s)
dPdz = -3.2  # pressure gradient (Pa/m)
h = 0.1  # domain height (m)
n_x = n_y = 80  # number of grid points
dx = dy = h / (n_x - 1)  # grid spacing

# Initialize the velocity field
w = np.zeros((n_x, n_y))

# Coefficients for the FVM
a_E = a_W = mu * dy / dx
a_N = a_S = mu * dx / dy
a_P = a_E + a_W + a_N + a_S
S_u = dPdz * dx * dy

# Jacobi iteration parameters
tolerance = 1e-6
max_iterations = 10000

# Jacobi iteration
for iteration in range(max_iterations):
    w_old = w.copy()
    
    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            w[i, j] = (a_W * w_old[i-1, j] + a_E * w_old[i+1, j] +
                       a_S * w_old[i, j-1] + a_N * w_old[i, j+1] +
                       S_u) / a_P
    
    # Check for convergence
    if np.linalg.norm(w - w_old, ord=np.inf) < tolerance:
        print(f"Converged after {iteration} iterations")
        break
else:
    print("Did not converge within the maximum number of iterations")

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/w_2D_Diffusion_FVM.npy', w)

# Visualize the final solution
plt.contourf(w, levels=50, cmap='viridis')
plt.colorbar(label='Velocity w (m/s)')
plt.title('Velocity Contour Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()