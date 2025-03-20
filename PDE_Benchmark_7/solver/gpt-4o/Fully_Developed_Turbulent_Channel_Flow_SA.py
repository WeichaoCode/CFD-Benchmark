import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

class Mesh:
    def __init__(self, n, H):
        self.n = n
        self.H = H
        self.y = self.generate_non_uniform_mesh()
        self.dy = np.diff(self.y)
        self.D1 = self.first_derivative_matrix()
        self.D2 = self.second_derivative_matrix()

    def generate_non_uniform_mesh(self):
        # Generate a non-uniform mesh clustered near the walls using a hyperbolic tangent
        y = -np.cosh(np.linspace(0, np.arccosh(10), self.n - 2)) + 1
        y = (y - y.min()) / (y.max() - y.min()) * self.H
        return np.concatenate(([0], y, [self.H]))

    def first_derivative_matrix(self):
        # Central difference scheme for first derivative matrix
        D = np.zeros((self.n, self.n))
        for i in range(1, self.n - 1):
            D[i, i - 1] = -1 / (self.dy[i - 1] + self.dy[i])
            D[i, i + 1] = 1 / (self.dy[i - 1] + self.dy[i])
        return D

    def second_derivative_matrix(self):
        # Central difference scheme for second derivative matrix
        D = np.zeros((self.n, self.n))
        for i in range(1, self.n - 1):
            D[i, i - 1] = 2 / (self.dy[i - 1] * (self.dy[i - 1] + self.dy[i]))
            D[i, i] = -2 / (self.dy[i - 1] * self.dy[i])
            D[i, i + 1] = 2 / (self.dy[i] * (self.dy[i - 1] + self.dy[i]))
        return D

def compute_turbulent_viscosity(tilde_nu, rho, nu):
    chi = tilde_nu / nu
    fv1 = chi**3 / (chi**3 + 7.1**3)
    mu_t = rho * tilde_nu * fv1
    return mu_t

def solve_spalart_allmaras(mesh, Re_tau, rho):
    # Initialize tilde_nu and parameters for Spalart-Allmaras model
    c_b1 = 0.1355; c_b2 = 0.622; c_b3 = 2/3
    kappa = 0.41; c_w1 = c_b1 / kappa**2 + (1.0 + c_b2) / c_b3
    nu = 1 / Re_tau
    tilde_nu = np.zeros(mesh.n)
        
    # Iterative solver for tilde_nu using a simplified form of the SA model
    # This step assumes some iterative method to solve for tilde_nu
    # Simple dummy assignment; this should be replaced by the actual computation
    tilde_nu[:] = 0.1 * nu  # Placeholder value for demonstration purposes
    
    mu_t = compute_turbulent_viscosity(tilde_nu, rho, nu)
    return mu_t

def solve_linear_system(D1, D2, mu_eff, mesh, b):
    # Construct the coefficient matrix A using effective viscosity
    A = np.zeros((mesh.n, mesh.n))
    A[0, 0] = 1.0  # Boundary condition at the wall
    A[-1, -1] = 1.0  # Boundary condition at the wall
    for i in range(1, mesh.n - 1):
        A[i, i - 1] = D1[i, i - 1] * (mu_eff[i] + mu_eff[i - 1]) / 2 + D2[i, i - 1] * mu_eff[i]
        A[i, i] = - (D1[i, i] * (mu_eff[i + 1] + mu_eff[i - 1]) / 2 + D2[i, i] * mu_eff[i])
        A[i, i + 1] = D1[i, i + 1] * (mu_eff[i] + mu_eff[i + 1]) / 2 + D2[i, i + 1] * mu_eff[i]

    # Solve the linear system using LU decomposition
    lu, piv = lu_factor(A)
    u = lu_solve((lu, piv), b)
    return u

def main():
    # User-defined inputs
    Re_tau = 395
    rho = 1.0
    mu = 1 / Re_tau
    n = 100
    H = 2
    
    # Initialize mesh
    mesh = Mesh(n, H)
    
    # Solve the Spalart-Allmaras model for mu_t
    mu_t = solve_spalart_allmaras(mesh, Re_tau, rho)
    
    # Compute effective viscosity
    mu_eff = mu + mu_t

    # Discretize the governing equation and solve the system
    b = np.zeros(n)
    b[1:-1] = -1  # Source term
    u = solve_linear_system(mesh.D1, mesh.D2, mu_eff, mesh, b)

    # Save velocity profile
    np.save('velocity_profile.npy', u)
    
    # Plot velocity distribution
    plt.plot(u, mesh.y, label='Turbulent Profile')
    y_lam = np.linspace(0, H, n)
    u_lam = 4 * (1 - (y_lam / H)**2)  # Parabolic laminar profile
    plt.plot(u_lam, y_lam, label='Laminar Profile', linestyle='--')
    plt.xlabel('Velocity')
    plt.ylabel('y')
    plt.title('Velocity Profile in a Turbulent Channel Flow')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()