import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

# Define constants
Re_tau = 395
rho = 1.0
mu = 1/Re_tau
n = 100
H = 2

# Define Mesh class
class Mesh:
    def __init__(self, n, H):
        self.n = n
        self.H = H
        self.y = self.compute_y()
        self.dy = self.compute_dy()
        self.d2y = self.compute_d2y()

    def compute_y(self):
        # Compute y-direction mesh points
        # Use a stretching function to cluster points near the walls
        return np.linspace(0, self.H, self.n)

    def compute_dy(self):
        # Compute first derivative matrix
        return np.gradient(self.y, edge_order=2)

    def compute_d2y(self):
        # Compute second derivative matrix
        return np.gradient(self.dy, edge_order=2)

# Define function to compute turbulent viscosity
def compute_turbulent_viscosity(rho, k, omega, S, F2, a1):
    return rho * k * min(1/omega, a1/(np.abs(S)*F2))

# Define function to solve linear system
def solve_linear_system(A, b):
    # Use LU decomposition to solve the system
    lu, piv = lu_factor(A)
    return lu_solve((lu, piv), b)

# Define function to compute velocity profile
def compute_velocity_profile(mesh, rho, mu):
    # Initialize velocity profile
    u = np.zeros(mesh.n)

    # Compute turbulent kinetic energy and dissipation
    k = np.zeros(mesh.n)
    omega = np.zeros(mesh.n)

    # Discretize the governing equation using finite difference method
    A = np.zeros((mesh.n, mesh.n))
    b = np.zeros(mesh.n)

    # Solve the linear system
    u = solve_linear_system(A, b)

    return u

# Create mesh
mesh = Mesh(n, H)

# Compute velocity profile
u = compute_velocity_profile(mesh, rho, mu)

# Save the computed velocity profile
np.save('velocity_profile.npy', u)

# Plot the velocity profile
plt.plot(u, mesh.y)
plt.xlabel('Velocity')
plt.ylabel('y')
plt.title('Velocity Profile')
plt.show()