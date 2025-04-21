import numpy as np

def solve_lane_emden(n=3.0, r_max=1.0, num_points=100):
    """
    Solves the Lane-Emden equation using a finite difference method.

    Args:
        n (float): Polytropic index.
        r_max (float): Outer radius.
        num_points (int): Number of radial points.

    Returns:
        numpy.ndarray: Solution f(r) at the final time step.
    """

    # Discretization
    r = np.linspace(0, r_max, num_points)
    dr = r[1] - r[0]

    # Initial guess
    R0 = 5
    f = R0**(2/(n-1)) * (1 - r**2)**2

    # Boundary conditions
    f[-1] = 0.0

    # Iterative solution (Newton-Raphson)
    tolerance = 1e-6
    max_iterations = 1000
    
    for _ in range(max_iterations):
        f_old = f.copy()

        # Construct the Jacobian matrix and residual vector
        J = np.zeros((num_points, num_points))
        residual = np.zeros(num_points)

        # Interior points
        for i in range(1, num_points - 1):
            J[i, i-1] = 1 - dr / (2 * r[i])
            J[i, i] = -2 + dr**2 * n * f[i]**(n-1)
            J[i, i+1] = 1 + dr / (2 * r[i])
            residual[i] = (f[i-1] - 2*f[i] + f[i+1]) / dr**2 + (f[i+1] - f[i-1]) / (2 * r[i] * dr) + f[i]**n

        # Boundary condition at r = r_max
        J[-1, -1] = 1
        residual[-1] = f[-1]

        # Regularity condition at r = 0 (L'Hopital's rule)
        J[0, 0] = -2 + dr**2 * n * f[0]**(n-1)
        J[0, 1] = 2
        residual[0] = 2 * (f[1] - f[0]) / dr**2 + f[0]**n

        # Solve the linear system
        delta_f = np.linalg.solve(J, -residual)
        f += delta_f

        # Check for convergence
        if np.max(np.abs(f - f_old)) < tolerance:
            break

    return f

if __name__ == "__main__":
    # Solve the Lane-Emden equation
    f_solution = solve_lane_emden()

    # Save the solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/f_solution_Lane_Emden_Equation.npy', f_solution)