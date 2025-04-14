import numpy as np

def solve_poisson(Lx, Ly, nx, ny):
    """
    Solves the Poisson equation with Dirichlet boundary conditions using the finite difference method.

    Args:
        Lx (float): Length of the domain in the x-direction.
        Ly (float): Length of the domain in the y-direction.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.

    Returns:
        p (numpy.ndarray): The solution to the Poisson equation.
    """

    # Domain
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Initialize solution and source term
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    # Define source term
    x_center1 = Lx / 4
    y_center1 = Ly / 4
    x_center2 = 3 * Lx / 4
    y_center2 = 3 * Ly / 4

    # Find the indices closest to the center points
    x_index1 = np.argmin(np.abs(x - x_center1))
    y_index1 = np.argmin(np.abs(y - y_center1))
    x_index2 = np.argmin(np.abs(x - x_center2))
    y_index2 = np.argmin(np.abs(y - y_center2))

    b[y_index1, x_index1] = 100
    b[y_index2, x_index2] = -100

    # Boundary conditions
    p[0, :] = 0  # y = 0
    p[-1, :] = 0  # y = Ly
    p[:, 0] = 0  # x = 0
    p[:, -1] = 0  # x = Lx

    # Iterative solution (Jacobi method)
    max_iter = 10000
    tolerance = 1e-6
    error = 1.0
    iter_count = 0

    while error > tolerance and iter_count < max_iter:
        p_old = np.copy(p)

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                p[i, j] = 0.25 * (p_old[i+1, j] + p_old[i-1, j] + p_old[i, j+1] + p_old[i, j-1] - dx**2 * b[i, j])

        # Apply boundary conditions again to ensure they are satisfied
        p[0, :] = 0  # y = 0
        p[-1, :] = 0  # y = Ly
        p[:, 0] = 0  # x = 0
        p[:, -1] = 0  # x = Lx

        error = np.max(np.abs(p - p_old))
        iter_count += 1

    return p

if __name__ == "__main__":
    # Problem parameters
    Lx = 2.0
    Ly = 1.0
    nx = 50
    ny = 50

    # Solve the Poisson equation
    p = solve_poisson(Lx, Ly, nx, ny)

    # Save the solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Poisson_Equation.npy', p)