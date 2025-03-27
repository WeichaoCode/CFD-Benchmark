import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

# Constants
C_mu = 0.09
C_e1 = 1.44
C_e2 = 1.92
sigma_k = 1.0
sigma_epsilon = 1.3
Re_tau = 395
rho = 1.0
mu = 1.0 / Re_tau
H = 2.0

# Mesh class for generating y-direction mesh points
class Mesh:
    def __init__(self, n, H):
        self.n = n
        self.H = H
        self.dy = self.H / (self.n - 1)
        self.y = self.create_mesh()
        self.dydy, self.d2ydy2 = self.create_finite_difference_matrices()

    def create_mesh(self):
        # Cluster mesh points near the walls using hyperbolic stretching
        y = np.sinh(np.linspace(-1, 1, self.n)) / np.sinh(1)
        return 0.5 * (y + 1) * self.H  # Rescale to [0, H]

    def create_finite_difference_matrices(self):
        # Central difference first and second derivative matrices
        d = -2 * np.eye(self.n) + np.eye(self.n, k=1) + np.eye(self.n, k=-1)
        dydy = np.zeros((self.n, self.n))
        d2ydy2 = np.zeros((self.n, self.n))

        for i in range(1, self.n - 1):
            dydy[i, i - 1] = 1 / self.dy
            dydy[i, i + 1] = -1 / self.dy
            d2ydy2[i, i] = -2 / (self.dy ** 2)
            d2ydy2[i, i - 1] = 1 / (self.dy ** 2)
            d2ydy2[i, i + 1] = 1 / (self.dy ** 2)

        return dydy, d2ydy2

# Function to compute turbulent viscosity
def compute_turbulent_viscosity(k, epsilon):
    return C_mu * rho * (k**2 / epsilon)

# Function to calculate k and epsilon depending on the flow conditions
def compute_k_epsilon(n, mesh):
    k = np.ones(n) * 0.1  # Initial guess for k
    epsilon = np.ones(n) * 0.1  # Initial guess for epsilon
    P_k = np.zeros(n)
    
    for it in range(1000):  # Iterate until convergence
        mu_t = compute_turbulent_viscosity(k, epsilon)
        
        # Model for turbulent production term
        P_k = mu_t * (mesh.dydy @ k)  # Placeholder for production term, adjust as necessary
        
        # k equation
        rhs_k = P_k - rho * epsilon
        k[1:-1] = np.linalg.solve(
            np.eye(n)[1:-1, 1:-1] - (mesh.d2ydy2[1:-1, 1:-1] @ (mu + mu_t[1:-1] / sigma_k)),
            rhs_k[1:-1]
        )
        
        # epsilon equation
        rhs_epsilon = (C_e1 * (P_k[1:-1] / k[1:-1]) * np.mean(k) - C_e2 * (epsilon[1:-1] / k[1:-1]) * epsilon[1:-1])
        epsilon[1:-1] = np.linalg.solve(
            np.eye(n)[1:-1, 1:-1] - (mesh.d2ydy2[1:-1, 1:-1] @ (mu + mu_t[1:-1] / sigma_epsilon)),
            rhs_epsilon
        )

    return k, epsilon

# Create Mesh
n = 100
mesh = Mesh(n, H)
# Compute k and epsilon
k, epsilon = compute_k_epsilon(n, mesh)

# Create linear system for solving the velocity profile
def compute_velocity_profile(k, epsilon):
    mu_t = compute_turbulent_viscosity(k, epsilon)
    A = np.eye(n) / (mu + mu_t / (H/2))
    b = np.zeros(n)
    b[1:-1] = 1 - (mu_t[1:-1] * (mesh.dydy[1:-1, :] @ k) / mu)
    
    # Solve linear system
    lu, piv = lu_factor(A)
    u = lu_solve((lu, piv), b)
    
    return u

# Compute velocity profile
velocity_profile = compute_velocity_profile(k, epsilon)

# Save the computed velocity profile in .npy format
np.save('velocity_profile.npy', velocity_profile)

# Plot the velocity profile
y_plot = mesh.y
plt.plot(velocity_profile, y_plot, label='Turbulent velocity profile')
plt.plot(1 - (y_plot / H)**2, y_plot, 'r--', label='Laminar profile')
plt.xlabel('Velocity')
plt.ylabel('y')
plt.title('Velocity Profile in the Channel')
plt.legend()
plt.grid()
plt.show()