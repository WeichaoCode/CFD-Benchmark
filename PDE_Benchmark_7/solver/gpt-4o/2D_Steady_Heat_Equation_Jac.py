import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 5.0, 4.0  # Domain dimensions
nx, ny = 50, 40  # Number of grid points
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
beta = dx / dy
max_iter = 10000  # Maximum number of iterations
tolerance = 1e-6  # Convergence criterion

# Dirichlet boundary conditions
T_left = 10.0
T_top = 0.0
T_right = 40.0
T_bottom = 20.0

# Grid initialization
T = np.zeros((nx, ny))

# Apply boundary conditions
T[0, :] = T_left
T[-1, :] = T_right
T[:, 0] = T_bottom
T[:, -1] = T_top

# Choose method: Jacobi, Gauss-Seidel, or SOR
method = 'Jacobi'  # Change to 'Jacobi' or 'Gauss-Seidel' if needed
omega = 1.5  # Over-relaxation factor for SOR


# Iterative solver
def solve_steady_heat_equation(T):
    for iter in range(max_iter):
        T_old = T.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if method == 'Jacobi':  # Point Jacobi method
                    T[i, j] = (T_old[i + 1, j] + T_old[i - 1, j] + beta ** 2 * (T_old[i, j + 1] + T_old[i, j - 1])) / (
                            2 * (1 + beta ** 2))
                elif method == 'Gauss-Seidel':  # Point Gauss-Seidel method
                    T[i, j] = (T[i + 1, j] + T[i - 1, j] + beta ** 2 * (T[i, j + 1] + T[i, j - 1])) / (
                            2 * (1 + beta ** 2))
                elif method == 'SOR':  # Point Successive Over-relaxation
                    T[i, j] = (1 - omega) * T[i, j] + omega * (
                            T[i + 1, j] + T[i - 1, j] + beta ** 2 * (T[i, j + 1] + T[i, j - 1])) / (
                                      2 * (1 + beta ** 2))

        # Convergence check
        residual = np.linalg.norm(T - T_old, ord='fro')
        if residual < tolerance:
            print(f'Converged after {iter} iterations with residual {residual:.2e}')
            break
    return T


# Solve the equation
T = solve_steady_heat_equation(T)

# Save the final temperature distribution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/T_2D_Steady_Heat_Equation_Jac.npy', T)

# Visualization
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, T.T, 50, cmap='hot')
plt.colorbar(cp)
plt.title('Steady-State Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().invert_yaxis()
plt.show()
