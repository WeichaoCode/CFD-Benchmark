import numpy as np
import matplotlib.pyplot as plt

def poisson_2d_solver(nx, ny, lx, ly, max_iter=500, tol=1e-5):
    # Define spatial step sizes
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    
    # Initialize the pressure field
    p = np.zeros((ny, nx))
    
    # Define the source term
    b = np.zeros((ny, nx))
    b[int(ny/4), int(nx/4)] = 100
    b[int(3*ny/4), int(3*nx/4)] = -100
    
    # Define the relaxation factor for SOR (1.0 for Jacobi)
    relaxation_factor = 1.0
    
    # Iterate to solve Poisson's equation
    for _ in range(max_iter):
        p_old = p.copy()
        
        # Update the pressure field using central differencing
        p[1:-1, 1:-1] = (
            ((p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 +
             (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2 -
             b[1:-1, 1:-1] * dx**2 * dy**2) 
            / (2 * (dx**2 + dy**2))
        )
        
        # Apply Dirichlet boundary conditions
        p[:, 0] = 0  # p = 0 at x = 0
        p[:, -1] = 0  # p = 0 at x = Lx
        p[0, :] = 0  # p = 0 at y = 0
        p[-1, :] = 0  # p = 0 at y = Ly

        # Check for convergence
        if np.linalg.norm(p - p_old, ord=2) < tol:
            print(f"Convergence reached after {_} iterations.")
            break
    
    return p

def plot_pressure_field(p, lx, ly):
    # Create mesh grid for plotting
    nx, ny = p.shape
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)

    # Plot the pressure field
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, p, alpha=0.75, cmap='viridis')
    plt.colorbar(cp)
    plt.title('Pressure Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Define domain parameters
Lx = 2.0
Ly = 1.0
nx = 50
ny = 50

# Solve the Poisson equation
p = poisson_2d_solver(nx, ny, Lx, Ly)

# Save the pressure field to a file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/p_2D_Possion.npy', p)

# Visualize the pressure field
plot_pressure_field(p, Lx, Ly)