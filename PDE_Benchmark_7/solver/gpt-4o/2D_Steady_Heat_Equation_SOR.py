import numpy as np
import matplotlib.pyplot as plt

# Define the computational domain and grid
Lx, Ly = 5.0, 4.0  # Domain size
dx, dy = 0.05, 0.05  # Grid spacing
beta = dx/dy

nx = int(Lx/dx) + 1  # Number of points in x
ny = int(Ly/dy) + 1  # Number of points in y

# Initialize temperature field
T = np.zeros((ny, nx))

# Boundary conditions
T[:, 0] = 10.0  # Left boundary (AB)
T[:, -1] = 40.0  # Right boundary (EF)
T[0, :] = 20.0  # Bottom boundary (G)
T[-1, :] = 0.0  # Top boundary (CD)

# SOR Method Parameters
omega = 1.5  # SOR relaxation factor, typically 1 < omega < 2
tolerance = 1e-5
max_iter = 5000

# Successive Over-Relaxation Method
def sor_solver(T, omega, beta, max_iter, tolerance):
    for iteration in range(max_iter):
        T_old = T.copy()
        
        # SOR Update
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                T[i, j] = (omega * 0.5 * ((T_old[i+1, j] + T[i-1, j]) + 
                                          beta**2 * (T[i, j+1] + T[i, j-1])) 
                           / (1 + beta**2) + (1 - omega) * T_old[i, j])
        
        # Convergence check
        res = np.linalg.norm(T - T_old, ord=np.inf)
        if res < tolerance:
            print(f'Converged after {iteration+1} iterations with residual {res:.2e}.')
            break
    else:
        print(f'SOR did not converge within the maximum number of iterations {max_iter}. Final residual = {res:.2e}.')

    return T

# Solve the steady-state heat distribution
T_final = sor_solver(T, omega, beta, max_iter, tolerance)

# Save the final temperature distribution
np.save('/PDE_Benchmark_7/results/prediction/T_2D_Steady_Heat_Equation_SOR.npy', T)

# Plot the temperature distribution
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, T_final, cmap='plasma', levels=50)
plt.colorbar(contour)
plt.title('Steady-State Temperature Distribution')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()