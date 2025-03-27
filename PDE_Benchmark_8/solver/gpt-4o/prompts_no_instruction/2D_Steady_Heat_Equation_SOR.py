import numpy as np

# Parameters
Lx, Ly = 5.0, 4.0  # Domain size
nx, ny = 101, 81   # Number of grid points
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # Grid spacing
tolerance = 1e-6   # Convergence tolerance
omega = 1.5        # Relaxation factor for SOR

# Initialize the temperature field
T = np.zeros((ny, nx))

# Apply boundary conditions
T[:, 0] = 10.0    # Left boundary
T[:, -1] = 40.0   # Right boundary
T[0, :] = 0.0     # Top boundary
T[-1, :] = 20.0   # Bottom boundary

# Successive Over-Relaxation (SOR) method
def solve_laplace_sor(T, dx, dy, omega, tolerance):
    max_iterations = 10000
    for iteration in range(max_iterations):
        T_old = T.copy()
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                T[j, i] = ((1 - omega) * T[j, i] +
                           omega * 0.5 * ((T[j, i+1] + T[j, i-1]) * dy**2 +
                                          (T[j+1, i] + T[j-1, i]) * dx**2) /
                           (dx**2 + dy**2))
        
        # Check for convergence
        diff = np.linalg.norm(T - T_old, ord=np.inf)
        if diff < tolerance:
            print(f'Converged after {iteration} iterations.')
            break
    else:
        print('Did not converge within the maximum number of iterations.')
    return T

# Solve the PDE
T_final = solve_laplace_sor(T, dx, dy, omega, tolerance)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/T_final_2D_Steady_Heat_Equation_SOR.npy', T_final)