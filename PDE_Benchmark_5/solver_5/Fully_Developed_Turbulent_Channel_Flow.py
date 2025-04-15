import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def main():
    # 1. Define parameters
    H = 1.0      # height
    ny = 100     # number of grid points in y
    mu = 1e-3    # viscosity
    mu_t = 0.01  # eddy viscosity

    # 2. Discretize the domain
    dy = H / (ny - 1)       # grid size
    y = np.linspace(0, H, ny) # y-grid

    # 3. Assemble the coefficient matrix A and right-hand side vector b
    diag_values = [(mu + mu_t) / dy**2, -2*(mu + mu_t) / dy**2, (mu + mu_t) / dy**2]
    A = diags(diag_values, [-1, 0, 1], shape=(ny, ny)).tocsc()
    A[0, 0], A[-1, -1] = 1, 1  # Set BCs
    b = -np.ones(ny)
    b[0], b[-1] = 0, 0         # Set BCs

    # 4. Solve for velocity
    u = spsolve(A, b)

    # 5. Visualize the velocity profile
    plt.figure()
    plt.plot(u, y)
    plt.xlabel('Velocity (u)')
    plt.ylabel('Height (y)')
    plt.title('Velocity profile in a turbulent channel')
    plt.grid(True)
    plt.show()

# Run the main function   
main()