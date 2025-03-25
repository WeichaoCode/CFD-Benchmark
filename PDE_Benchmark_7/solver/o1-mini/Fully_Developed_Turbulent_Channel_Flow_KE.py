import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
import numpy.linalg as la

class Mesh:
    def __init__(self, n, H, beta=1.5):
        self.n = n
        self.H = H
        self.beta = beta
        self.y = self.generate_mesh()
        self.dy = np.diff(self.y)
        self.D1, self.D2 = self.create_derivative_matrices()

    def generate_mesh(self):
        xi = np.linspace(0, 1, self.n)
        stretching = (np.tanh(self.beta * (xi - 0.5)) / np.tanh(self.beta / 2))
        y = (self.H / 2) * (1 + stretching)
        return y

    def create_derivative_matrices(self):
        n = self.n
        D1 = np.zeros((n, n))
        D2 = np.zeros((n, n))
        y = self.y

        for i in range(1, n-1):
            h0 = y[i] - y[i-1]
            h1 = y[i+1] - y[i]
            
            # First derivative coefficients
            D1[i, i-1] = -h1 / (h0 * (h0 + h1))
            D1[i, i] = (h1 - h0) / (h0 * h1)
            D1[i, i+1] = h0 / (h1 * (h0 + h1))
            
            # Second derivative coefficients
            D2[i, i-1] = 2.0 / (h0 * (h0 + h1))
            D2[i, i] = -2.0 / (h0 * h1)
            D2[i, i+1] = 2.0 / (h1 * (h0 + h1))

        # Boundary conditions (central differences)
        # First row (y=0)
        D1[0, 0] = -1.0 / y[1]
        D1[0, 1] = 1.0 / y[1]
        D2[0,0] = 2.0 / (y[1]**2)
        D2[0,1] = -2.0 / (y[1]**2)
        
        # Last row (y=H)
        D1[-1, -2] = -1.0 / (y[-1] - y[-2])
        D1[-1, -1] = 1.0 / (y[-1] - y[-2])
        D2[-1,-2] = 2.0 / ((y[-1] - y[-2])**2)
        D2[-1,-1] = -2.0 / ((y[-1] - y[-2])**2)
        
        return D1, D2

def compute_mu_t(C_mu, f_mu, rho, k, epsilon):
    return C_mu * f_mu * rho * k**2 / epsilon

def solve_linear_system(A, b):
    return solve(A, b)

def main():
    # User-Defined Inputs
    Re_tau = 395
    rho = 1.0
    mu = 1.0 / Re_tau
    H = 2.0
    n = 100

    # Model Constants
    C_mu = 0.09
    sigma_k = 1.0
    sigma_epsilon = 1.3
    C_e1 = 1.44
    C_e2 = 1.92

    # Initialize Mesh
    mesh = Mesh(n, H)
    y = mesh.y
    D1 = mesh.D1
    D2 = mesh.D2

    # Initialize variables
    k = np.full(n, 0.1)  # Initial guess for turbulent kinetic energy
    epsilon = np.full(n, 0.1)  # Initial guess for dissipation
    u = np.zeros(n)  # Velocity profile
    mu_t = compute_mu_t(C_mu, 1.0, rho, k, epsilon)

    # Iterative Solver Parameters
    max_iter = 1000
    tol = 1e-6
    alpha = 0.7  # Under-relaxation factor

    for it in range(max_iter):
        u_old = u.copy()
        k_old = k.copy()
        epsilon_old = epsilon.copy()
        mu_t_old = mu_t.copy()

        # Compute du/dy
        du_dy = D1 @ u

        # Compute turbulent production Pk
        Pk = mu_t * du_dy**2

        # Assemble k equation: (C_e1 f1 Pk - C_e2 f2 epsilon) * (epsilon/k) - d/dy [ (mu + mu_t/sigma_k ) dk/dy ] = 0
        # For simplicity, assume f1=f2=1
        A_k = np.zeros((n, n))
        b_k = Pk * C_e1 - C_e2 * epsilon

        # Diffusion term coefficients
        diffusion_k = mu + mu_t / sigma_k
        A_k = D2 * diffusion_k

        # Source term
        b_k += 0  # No source term apart from the above

        # RHS
        b_k = Pk * C_e1 - C_e2 * epsilon

        # Assemble epsilon equation
        # 0 = (C_e1 f1 Pk - C_e2 f2 epsilon) * (epsilon/k) + d/dy [ (mu + mu_t/sigma_epsilon) d epsilon/dy ]
        A_epsilon = np.zeros((n, n))
        diffusion_epsilon = mu + mu_t / sigma_epsilon
        A_epsilon = D2 * diffusion_epsilon

        # Source term
        b_epsilon = (C_e1 * Pk - C_e2 * epsilon) * epsilon / k

        # Solve for k
        try:
            k_new = solve_linear_system(A_k, b_k)
        except la.LinAlgError:
            print("Singular matrix encountered while solving for k.")
            break

        # Solve for epsilon
        try:
            epsilon_new = solve_linear_system(A_epsilon, b_epsilon)
        except la.LinAlgError:
            print("Singular matrix encountered while solving for epsilon.")
            break

        # Update mu_t
        mu_t_new = compute_mu_t(C_mu, 1.0, rho, k_new, epsilon_new)

        # Solve momentum equation: d/dy [ (mu + mu_t ) du/dy ] = 0
        diffusion_u = mu + mu_t_new
        A_u = D2 * diffusion_u
        b_u = np.zeros(n)  # Fully developed, no pressure gradient

        # Apply boundary conditions for velocity (no-slip)
        A_u[0,:] = 0
        A_u[0,0] = 1
        b_u[0] = 0

        A_u[-1,:] = 0
        A_u[-1,-1] = 1
        b_u[-1] = 0

        # Solve for u
        try:
            u_new = solve_linear_system(A_u, b_u)
        except la.LinAlgError:
            print("Singular matrix encountered while solving for u.")
            break

        # Under-relaxation
        k = alpha * k_new + (1 - alpha) * k
        epsilon = alpha * epsilon_new + (1 - alpha) * epsilon
        mu_t = alpha * mu_t_new + (1 - alpha) * mu_t
        u = alpha * u_new + (1 - alpha) * u

        # Check convergence
        res_u = np.linalg.norm(u - u_old, ord=np.inf)
        res_k = np.linalg.norm(k - k_old, ord=np.inf)
        res_epsilon = np.linalg.norm(epsilon - epsilon_old, ord=np.inf)

        if max(res_u, res_k, res_epsilon) < tol:
            print(f'Converged in {it+1} iterations.')
            break
    else:
        print('Did not converge within the maximum number of iterations.')

    # Save velocity profile
    np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_Fully_Developed_Turbulent_Channel_Flow_KE.npy', u)

    # Compute laminar parabolic profile
    y_centered = y - H/2
    u_laminar = (1/(2*mu)) * (mu * Re_tau / H) * (H**2 / 4 - y_centered**2)

    # Plotting
    plt.figure(figsize=(8,6))
    plt.plot(u, y, label='Turbulent Profile')
    plt.plot(u_laminar, y, '--', label='Laminar Profile')
    plt.xlabel('Velocity $u$')
    plt.ylabel('Channel Height $y$')
    plt.title('Velocity Profile in Channel Flow')
    plt.legend()
    plt.grid(True)
    plt.savefig('velocity_profile.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()