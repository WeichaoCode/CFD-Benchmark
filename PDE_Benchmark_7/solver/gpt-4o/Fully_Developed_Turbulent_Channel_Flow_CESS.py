import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

class Mesh:
    def __init__(self, n, H):
        self.n = n
        self.H = H
        self.y, self.dy = self.generate_mesh()
        self.D1, self.D2 = self.generate_derivative_matrices()

    def generate_mesh(self):
        # Generate non-uniform mesh with more points near the walls
        beta = 2.5  # Stretching factor
        y_stretched = np.linspace(0, 1, self.n)
        y = 0.5 * (1.0 - np.cos(np.pi * y_stretched)) * self.H
        dy = np.diff(y)
        return y, dy

    def generate_derivative_matrices(self):
        # First and second derivative matrices using central differences for the internal points
        D1 = np.zeros((self.n, self.n))
        D2 = np.zeros((self.n, self.n))

        for i in range(1, self.n - 1):
            D1[i, i - 1] = -1 / (2 * self.dy[i - 1])
            D1[i, i + 1] = 1 / (2 * self.dy[i])
            D2[i, i - 1] = 1 / self.dy[i - 1]**2
            D2[i, i] = -2 / (self.dy[i-1] * self.dy[i])
            D2[i, i + 1] = 1 / self.dy[i]**2

        # Boundary conditions using one-sided differences
        D1[0, 0] = -1 / self.dy[0]
        D1[0, 1] = 1 / self.dy[0]
        D1[-1, -2] = -1 / self.dy[-1]
        D1[-1, -1] = 1 / self.dy[-1]

        D2[0, 0] = D2[-1, -1] = 1
        D2[0, 1] = D2[-1, -2] = -2
        D2[0, 1] = D2[0, 2] = 1
        D2[-1, -3] = D2[-1, -2] = 1

        return D1, D2


def compute_turbulent_viscosity(y, Re_tau, H):
    # Constants
    kappa = 0.42
    A = 25.4

    y_plus = y * Re_tau / H
    mu_eff_over_mu = 0.5 * (1 + 1/9 * kappa**2 * Re_tau**2 * (2*y - y**2)**2 *
                            (3 - 4*y + 2*y**2)**2 * (1 - np.exp(-y_plus/A)))**0.5 - 0.5
    return mu_eff_over_mu


def solve_u(mesh, mu_eff, mu):
    n = mesh.n
    A = np.zeros((n, n))
    b = -np.ones(n)

    # Fill matrix A with finite difference approximations
    for i in range(1, n - 1):
        mu_eff_up = (mu_eff[i] + mu_eff[i + 1]) / 2
        mu_eff_down = (mu_eff[i] + mu_eff[i - 1]) / 2
        A[i, i - 1] = mu_eff_down / (mesh.dy[i - 1]**2)
        A[i, i] = -mu_eff_up / (mesh.dy[i]**2) - mu_eff_down / (mesh.dy[i - 1]**2)
        A[i, i + 1] = mu_eff_up / (mesh.dy[i]**2)

    # Boundary conditions
    A[0, 0] = 1
    b[0] = 0
    A[-1, -1] = 1
    b[-1] = 0

    # Solve the linear system
    lu, piv = lu_factor(A)
    u = lu_solve((lu, piv), b)
    return u

def main():
    # Parameters
    n = 100
    H = 2
    Re_tau = 395
    rho = 1.0
    mu = 1.0 / Re_tau

    # Generate mesh
    mesh = Mesh(n, H)

    # Compute effective viscosity
    mu_eff = mu * compute_turbulent_viscosity(mesh.y, Re_tau, H) + mu

    # Solve for velocity profile
    u = solve_u(mesh, mu_eff, mu)

    # Save the velocity profile
    np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/Fully_Developed_Turbulent_Channel_Flow_CESS.npy', u)

    # Plotting
    plt.figure()
    plt.plot(u, mesh.y, label='Turbulent flow')
    plt.plot(-1.5 * (mesh.y * (mesh.y - H)), mesh.y, label='Laminar flow', linestyle='--')
    plt.title('Velocity Profile')
    plt.xlabel('Velocity')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()