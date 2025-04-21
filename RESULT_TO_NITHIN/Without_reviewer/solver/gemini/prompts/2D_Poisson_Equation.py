import numpy as np

def solve_poisson(Lx, Ly, nx, ny, save_values):
    """
    Solves the Poisson equation using the finite difference method.

    Args:
        Lx (float): Length of the domain in the x-direction.
        Ly (float): Length of the domain in the y-direction.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        save_values (list): List of variable names to save.
    """

    # Domain
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Initialize p
    p = np.zeros((ny, nx))

    # Source term
    b = np.zeros((ny, nx))
    x_coords, y_coords = np.meshgrid(x, y)
    b[(abs(x_coords - Lx/4) < dx/2) & (abs(y_coords - Ly/4) < dy/2)] = 100
    b[(abs(x_coords - 3*Lx/4) < dx/2) & (abs(y_coords - 3*Ly/4) < dy/2)] = -100

    # Boundary conditions
    p[0, :] = 0
    p[-1, :] = 0
    p[:, 0] = 0
    p[:, -1] = 0

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

        # Boundary conditions (again, to enforce after each iteration)
        p[0, :] = 0
        p[-1, :] = 0
        p[:, 0] = 0
        p[:, -1] = 0

        error = np.max(np.abs(p - p_old))
        iter_count += 1

    # Save the solution
    if 'p' in save_values:
        np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Poisson_Equation.npy', p)

if __name__ == '__main__':
    Lx = 2.0
    Ly = 1.0
    nx = 50
    ny = 50
    save_values = ['p']
    solve_poisson(Lx, Ly, nx, ny, save_values)