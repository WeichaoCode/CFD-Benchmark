import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

class Mesh:
    def __init__(self, n, H):
        self.n = n
        self.H = H
        self.y = self.generate_mesh()
        self.delta_y = np.diff(self.y)
        self.mu_eff_interface = None

    def generate_mesh(self):
        s = np.linspace(0, 1, self.n)
        y = 0.5 * self.H * (1 - np.cos(np.pi * s))
        return y

def compute_mu_eff(y, Re_tau, kappa=0.42, A=25.4, mu=1/395):
    y_plus = y * Re_tau
    term = (1/9) * kappa**2 * Re_tau**2 * (2*y - y**2)**2 * (3 - 4*y + 2*y**2)**2 * (1 - np.exp(-y_plus / A))
    mu_eff_over_mu = 0.5 * (1 + np.sqrt(1 + term)) 
    return mu_eff_over_mu * mu

def assemble_system(mesh, mu_eff):
    n_internal = mesh.n - 2
    A = np.zeros((3, n_internal))
    b = -np.ones(n_internal)

    for i in range(1, mesh.n-1):
        idx = i - 1
        dy_minus = mesh.y[i] - mesh.y[i-1]
        dy_plus = mesh.y[i+1] - mesh.y[i]
        
        mu_minus = 0.5 * (mu_eff[i] + mu_eff[i-1])
        mu_plus = 0.5 * (mu_eff[i] + mu_eff[i+1])
        
        A[0, idx] = mu_plus / dy_plus**2  # Upper diagonal
        A[1, idx] = -(mu_plus + mu_minus) / dy_plus / dy_minus  # Main diagonal
        A[2, idx] = mu_minus / dy_minus**2  # Lower diagonal

    return A, b

def solve_velocity(n, H, Re_tau, rho, mu):
    mesh = Mesh(n, H)
    mu_eff = compute_mu_eff(mesh.y, Re_tau)
    A_matrix, b_vector = assemble_system(mesh, mu_eff)
    
    # Since A is tridiagonal, use solve_banded
    ab = np.zeros((3, n-2))
    ab[0,1:] = A_matrix[0, :-1]  # Upper diagonal
    ab[1, :] = A_matrix[1, :]     # Main diagonal
    ab[2, :-1] = A_matrix[2, 1:]  # Lower diagonal
    u_internal = solve_banded((1,1), ab, b_vector)
    
    # Assemble full velocity including boundary conditions
    u = np.zeros(n)
    u[1:-1] = u_internal
    return mesh.y, u

def plot_velocity(y, u, H):
    # Laminar profile
    u_laminar = 1 - (2*(y/H - 0.5))**2
    plt.figure(figsize=(8,6))
    plt.plot(u, y, label='Turbulent Velocity Profile')
    plt.plot(u_laminar, y, '--', label='Laminar Velocity Profile')
    plt.xlabel('Velocity u')
    plt.ylabel('y')
    plt.title('Velocity Profile in Channel Flow')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # User-Defined Inputs
    Re_tau = 395
    rho = 1.0
    mu = 1 / Re_tau
    n = 100
    H = 2

    y, u = solve_velocity(n, H, Re_tau, rho, mu)
    plot_velocity(y, u, H)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_Fully_Developed_Turbulent_Channel_Flow_CESS.npy', u)

if __name__ == "__main__":
    main()