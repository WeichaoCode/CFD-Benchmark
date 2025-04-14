import numpy as np

def solve_laplace(nx, ny, x_max, y_max):
    """
    Solves the Laplace equation on a 2D domain with specified boundary conditions.

    Args:
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        x_max (float): Maximum value of x.
        y_max (float): Maximum value of y.

    Returns:
        numpy.ndarray: The solution p(x, y) at the final iteration.
    """

    # Domain parameters
    x = np.linspace(0, x_max, nx)
    y = np.linspace(0, y_max, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Initialize p
    p = np.zeros((ny, nx))

    # Boundary conditions
    p[:, 0] = 0  # Left boundary: p = 0
    p[:, -1] = y  # Right boundary: p = y

    # Iterative solution (Jacobi method)
    max_iter = 10000
    tolerance = 1.0e-6
    error = 1.0
    iter_count = 0

    while error > tolerance and iter_count < max_iter:
        p_old = np.copy(p)

        # Update interior points
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                p[i, j] = 0.25 * (p_old[i+1, j] + p_old[i-1, j] + p_old[i, j+1] + p_old[i, j-1])

        # Neumann boundary conditions (top and bottom)
        p[0, :] = p[1, :]  # Bottom boundary: dp/dy = 0
        p[-1, :] = p[-2, :]  # Top boundary: dp/dy = 0

        # Calculate error
        error = np.max(np.abs(p - p_old))
        iter_count += 1

    return p

if __name__ == '__main__':
    # Problem parameters
    nx = 50
    ny = 25
    x_max = 2.0
    y_max = 1.0

    # Solve the Laplace equation
    p = solve_laplace(nx, ny, x_max, y_max)

    # Save the solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Laplace_Equation.npy', p)