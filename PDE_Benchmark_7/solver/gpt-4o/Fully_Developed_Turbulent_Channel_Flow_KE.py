import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.sparse import diags

class Mesh:
    def __init__(self, H, n):
        self.H = H
        self.n = n
        self.y, self.dy = self.generate_non_uniform_grid()
        self.dydy = self.dy ** 2
        self.Dy, self.Dy2 = self.compute_finite_difference_matrices()

    def generate_non_uniform_grid(self):
        # Use a hyperbolic tangent distribution to cluster points near the walls
        beta = 2.0
        y = np.linspace(0, 1, self.n)
        y_non_uniform = 0.5 * (1 - np.cos(y * np.pi))  # Double cosine stretching
        return self.H * y_non_uniform, np.gradient(self.H * y_non_uniform)
    
    def compute_finite_difference_matrices(self):
        # First derivative matrix
        Dy = diags([-1, 0, 1], [-1, 0, 1], shape=(self.n, self.n)).toarray()
        Dy[0, 0:2] = [-1, 1]  # Forward difference at the first point
        Dy[-1, -2:] = [-1, 1]  # Backward difference at the last point
        Dy /= 2 * self.dy  # Centers difference except at boundaries

        # Second derivative matrix
        Dy2 = diags([1, -2, 1], [-1, 0, 1], shape=(self.n, self.n)).toarray()
        Dy2[0, 0] = Dy2[-1, -1] = 1

        Dy2 /= self.dydy

        return Dy, Dy2

def compute_turbulent_viscosity(k, epsilon, rho, C_mu=0.09, f_mu=1.0):
    return C_mu * f_mu * rho * (k ** 2 / epsilon)

def solve_linear_system(A, b):
    # Solve the linear system Ax = b
    return solve(A, b)

def main():
    # User-defined parameters
    Re_tau = 395
    rho = 1.0
    mu = 1 / Re_tau
    H = 2
    n = 100
    C_mu = 0.09
    C_e1, C_e2 = 1.5, 1.83
    sigma_k, sigma_eps = 1.0, 1.3

    # Generate mesh
    mesh = Mesh(H, n)

    # Example k and epsilon initial values
    k = np.ones(n) * 1.5
    epsilon = np.ones(n) * 1.5

    # Compute turbulent viscosity
    mu_t = compute_turbulent_viscosity(k, epsilon, rho, C_mu)

    # Initialize arrays
    u = np.zeros(n)
    b = np.zeros(n)

    # Construct A matrix
    A = np.eye(n)
    # Example: This section needs to be implemented according to the problem-specific linear system

    # Solve the matrix system
    u = solve_linear_system(A, b)

    # Plot results
    plt.plot(u, mesh.y, label='Turbulent Vel. Profile')
    y_laminar = np.linspace(0, H, 100)
    u_laminar = 1 - (y_laminar - H / 2) ** 2
    plt.plot(u_laminar, y_laminar, label='Laminar Profile')
    plt.xlabel("Velocity (u)")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.title("Velocity Profile in Channel Flow")
    plt.show()

    # Save to .npy file
    np.save('velocity_profile.npy', u)

if __name__ == "__main__":
    main()