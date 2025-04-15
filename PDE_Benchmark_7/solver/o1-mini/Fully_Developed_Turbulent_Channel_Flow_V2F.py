import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# -----------------------------
# Mesh Generation Class
# -----------------------------
class Mesh:
    def __init__(self, n, H, stretching='cosine'):
        self.n = n
        self.H = H
        self.stretching = stretching
        self.y = self.generate_mesh()
        self.dy = np.diff(self.y)
        self.D1, self.D2 = self.build_derivative_matrices()

    def generate_mesh(self):
        # Generate a non-uniform mesh clustered near the walls
        xi = np.linspace(0, 1, self.n)
        if self.stretching == 'cosine':
            # Cosine stretching
            y = self.H / 2 * (1 - np.cos(np.pi * xi))
        elif self.stretching == 'tanh':
            # Hyperbolic tangent stretching
            beta = 1.5
            y = self.H / 2 * (1 + np.tanh(beta * (xi - 0.5)) / np.tanh(beta / 2))
        else:
            # Uniform mesh
            y = self.H * xi
        return y

    def build_derivative_matrices(self):
        D1 = np.zeros((self.n, self.n))
        D2 = np.zeros((self.n, self.n))
        
        for i in range(1, self.n-1):
            dy_minus = self.y[i] - self.y[i-1]
            dy_plus = self.y[i+1] - self.y[i]
            dy_total = dy_minus + dy_plus

            # First derivative (central difference)
            D1[i, i-1] = -dy_plus / dy_total
            D1[i, i+1] = dy_minus / dy_total
            D1[i, i] = 0.0

            # Second derivative
            D2[i, i-1] = 2.0 / (dy_minus * (dy_minus + dy_plus))
            D2[i, i] = -2.0 / (dy_minus * dy_plus)
            D2[i, i+1] = 2.0 / (dy_plus * (dy_minus + dy_plus))
        
        # Boundary conditions for first derivative (one-sided)
        D1[0,0] = -1.5 / self.dy[0]
        D1[0,1] = 2.0 / self.dy[0]
        D1[0,2] = -0.5 / self.dy[0]

        D1[-1,-1] = 1.5 / self.dy[-1]
        D1[-1,-2] = -2.0 / self.dy[-1]
        D1[-1,-3] = 0.5 / self.dy[-1]

        # Boundary conditions for second derivative (Neumann)
        D2[0,0] = 1.0 / self.dy[0]**2
        D2[0,1] = -2.0 / self.dy[0]**2
        D2[0,2] = 1.0 / self.dy[0]**2

        D2[-1,-1] = 1.0 / self.dy[-1]**2
        D2[-1,-2] = -2.0 / self.dy[-1]**2
        D2[-1,-3] = 1.0 / self.dy[-1]**2

        return D1, D2

# -----------------------------
# Constants and Parameters
# -----------------------------
# User-Defined Inputs
Re_tau = 395
rho = 1.0
mu = 1.0 / Re_tau  # Dynamic viscosity

# V2F Model Constants
C_mu = 0.09
C_e1 = 1.44
C_e2 = 1.92
sigma_k = 1.0
sigma_epsilon = 1.3
C1 = 2.0
C2 = 2.0

# Characteristic Length and Time
H = 2.0
L = H
T = 1.0  # Assuming non-dimensional for steady state

# Relaxation Factors
relax = 0.3

# Convergence Criteria
tolerance = 1e-6
max_iterations = 1000

# -----------------------------
# Initialize Mesh
# -----------------------------
n = 100
mesh = Mesh(n, H, stretching='cosine')
y = mesh.y

# -----------------------------
# Initialize Variables
# -----------------------------
k = np.full(n, 1e-3)      # Turbulent kinetic energy
epsilon = np.full(n, 1e-3)  # Turbulent dissipation
v2 = np.full(n, 1.0)      # Wall-normal fluctuation component
f = np.full(n, 1.0)       # Elliptic relaxation function

# Initialize mu_t
mu_t = np.zeros(n)

# Initialize Production term Pk
Pk = np.ones(n)

# Initialize Velocity
u = np.zeros(n)

# -----------------------------
# Assemble Boundary Conditions
# -----------------------------
def apply_boundary_conditions(A, b):
    # Apply Neumann BC for derivatives at walls
    # For k, epsilon, v2, f: typically zero gradient at center
    # No-slip for velocity: u=0 at walls
    # Symmetry at center: du/dy=0

    # Example for k
    A[0,:] = 0
    A[0,0] = 1
    b[0] = 1e-6  # Small value to avoid zero

    A[-1,:] = 0
    A[-1,-1] = 1
    b[-1] = 1e-6

    return A, b

# -----------------------------
# Main Iterative Solver
# -----------------------------
for it in range(max_iterations):
    k_old = k.copy()
    epsilon_old = epsilon.copy()
    v2_old = v2.copy()
    f_old = f.copy()
    mu_t_old = mu_t.copy()
    u_old = u.copy()

    # -------------------------
    # Update Eddy Viscosity
    # -------------------------
    mu_t = C_mu * rho * (epsilon / k)**0.5 * L  # Assuming T_t = L

    # -------------------------
    # Compute Production Term Pk
    # -------------------------
    # Compute du/dy using finite differences
    D1 = mesh.D1
    du_dy = D1 @ u
    Pk = mu_t * (du_dy)**2

    # -------------------------
    # Assemble and Solve for k
    # -------------------------
    # 0 = Pk - rho * epsilon + d/dy [ (mu + mu_t / sigma_k ) dk/dy ]

    A_k = np.zeros((n, n))
    b_k = Pk - rho * epsilon

    # Diffusion coefficient
    D_k = mu + mu_t / sigma_k

    for i in range(1, n-1):
        A_k[i,i-1] = D_k[i-1] * mesh.D1[i,i-1]
        A_k[i,i] = - (D_k[i-1] * mesh.D1[i,i-1] + D_k[i] * mesh.D1[i,i+1])
        A_k[i,i+1] = D_k[i] * mesh.D1[i,i+1]

    # Boundary conditions for k
    A_k, b_k = apply_boundary_conditions(A_k, b_k)

    # Solve for k
    k_new = solve(A_k, b_k)
    k = relax * k_new + (1 - relax) * k_old

    # -------------------------
    # Assemble and Solve for epsilon
    # -------------------------
    # 0 = (1/T)(C_e1 * Pk - C_e2 * rho * epsilon) + d/dy [ (mu + mu_t / sigma_epsilon ) depsilon/dy ]

    A_e = np.zeros((n, n))
    b_e = (C_e1 * Pk - C_e2 * rho * epsilon) / T

    D_epsilon = mu + mu_t / sigma_epsilon

    for i in range(1, n-1):
        A_e[i,i-1] = D_epsilon[i-1] * mesh.D1[i,i-1]
        A_e[i,i] = - (D_epsilon[i-1] * mesh.D1[i,i-1] + D_epsilon[i] * mesh.D1[i,i+1])
        A_e[i,i+1] = D_epsilon[i] * mesh.D1[i,i+1]

    # Boundary conditions for epsilon
    A_e, b_e = apply_boundary_conditions(A_e, b_e)

    # Solve for epsilon
    epsilon_new = solve(A_e, b_e)
    epsilon = relax * epsilon_new + (1 - relax) * epsilon_old

    # -------------------------
    # Assemble and Solve for v2
    # -------------------------
    # 0 = rho * k * f - 6 * rho * v2 * (epsilon / k) + d/dy [ (mu + mu_t / sigma_k ) dv2/dy ]

    A_v2 = np.zeros((n, n))
    b_v2 = rho * k * f - 6 * rho * v2 * (epsilon / k)

    D_v2 = mu + mu_t / sigma_k

    for i in range(1, n-1):
        A_v2[i,i-1] = D_v2[i-1] * mesh.D1[i,i-1]
        A_v2[i,i] = - (D_v2[i-1] * mesh.D1[i,i-1] + D_v2[i] * mesh.D1[i,i+1])
        A_v2[i,i+1] = D_v2[i] * mesh.D1[i,i+1]

    # Boundary conditions for v2
    A_v2, b_v2 = apply_boundary_conditions(A_v2, b_v2)

    # Solve for v2
    v2_new = solve(A_v2, b_v2)
    v2 = relax * v2_new + (1 - relax) * v2_old

    # -------------------------
    # Assemble and Solve for f
    # -------------------------
    # L^2 d2f/dy2 - f = [1/T (C1 (6 - v2) - (2/3)(C1 -1)) ] - C2 * Pk

    A_f = L**2 * mesh.D2 - np.diag(np.ones(n))
    b_f = (1.0 / T) * (C1 * (6.0 - v2) - (2.0/3.0)*(C1 -1.0)) - C2 * Pk

    # Boundary conditions for f (Neumann: df/dy = 0 at center, Dirichlet at walls)
    A_f, b_f = apply_boundary_conditions(A_f, b_f)

    # Solve for f
    f_new = solve(A_f, b_f)
    f = relax * f_new + (1 - relax) * f_old

    # -------------------------
    # Update Eddy Viscosity
    # -------------------------
    mu_t = C_mu * rho * np.sqrt(epsilon / k) * L

    # -------------------------
    # Assemble and Solve for Velocity u
    # -------------------------
    # d/dy [ (mu + mu_t ) du/dy ] = 0
    # Integrate once: (mu + mu_t) du/dy = tau (constant)
    # Apply boundary conditions u=0 at walls

    A_u = np.zeros((n, n))
    b_u = np.zeros(n)

    mu_eff = mu + mu_t

    for i in range(1, n-1):
        A_u[i,i-1] = mu_eff[i-1] / mesh.dy[i-1]**2
        A_u[i,i] = - (mu_eff[i-1] + mu_eff[i]) / mesh.dy[i-1]**2
        A_u[i,i+1] = mu_eff[i] / mesh.dy[i-1]**2

    # Boundary conditions for velocity
    A_u, b_u = apply_boundary_conditions(A_u, b_u)
    b_u[:] = 0.0  # No external forcing

    # Solve for u
    u_new = solve(A_u, b_u)
    u = relax * u_new + (1 - relax) * u_old

    # -------------------------
    # Check for Convergence
    # -------------------------
    diff = max(
        np.max(np.abs(k - k_old)),
        np.max(np.abs(epsilon - epsilon_old)),
        np.max(np.abs(v2 - v2_old)),
        np.max(np.abs(f - f_old)),
        np.max(np.abs(mu_t - mu_t_old)),
        np.max(np.abs(u - u_old))
    )

    if diff < tolerance:
        print(f'Converged in {it+1} iterations.')
        break
else:
    print('Did not converge within the maximum number of iterations.')

# -----------------------------
# Compute Laminar Profile for Comparison
# -----------------------------
# For laminar fully-developed channel flow: u(y) = (1/(2*mu)) * tau_w * y * (H - y)

# Assume tau_w = mu * Re_tau
tau_w = mu * Re_tau
u_lam = (tau_w / (2 * mu)) * y * (H - y)

# -----------------------------
# Plot Velocity Profiles
# -----------------------------
plt.figure(figsize=(8,6))
plt.plot(u, y, label='Turbulent Velocity Profile')
plt.plot(u_lam, y, '--', label='Laminar Velocity Profile')
plt.xlabel('Velocity u')
plt.ylabel('Channel Height y')
plt.title('Velocity Profile in Channel Flow')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Save Velocity Profile
# -----------------------------
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_Fully_Developed_Turbulent_Channel_Flow_V2F.npy', u)