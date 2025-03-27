import numpy as np

# Parameters
nx, ny = 31, 31
dx = 2 / (nx - 1)
dy = 1 / (ny - 1)
tolerance = 1e-5
max_iterations = 10000

# Initialize the potential field
p = np.zeros((ny, nx))

# Boundary conditions
p[:, 0] = 0  # Left boundary
p[:, -1] = np.linspace(0, 1, ny)  # Right boundary

# Iterative solver using Jacobi method
def solve_laplace(p, dx, dy, tolerance, max_iterations):
    pn = np.empty_like(p)
    for it in range(max_iterations):
        pn[:] = p[:]
        # Update the potential field
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]) +
                          dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1])) /
                         (2 * (dx**2 + dy**2)))

        # Neumann boundary conditions (top and bottom)
        p[0, :] = p[1, :]  # Top boundary
        p[-1, :] = p[-2, :]  # Bottom boundary

        # Check for convergence
        if np.max(np.abs(p - pn)) < tolerance:
            print(f'Converged after {it} iterations')
            break
    else:
        print('Did not converge within the maximum number of iterations')
    return p

# Solve the PDE
p_final = solve_laplace(p, dx, dy, tolerance, max_iterations)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_1/p_final_2D_Laplace_Equation.npy', p_final)