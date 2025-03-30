import numpy as np
from numpy.linalg import solve

# Parameters
H = 2.0
n = 100
mu = 1.0
rho = 1.0
B = 26.3
C = 0.14
D = 3.0
tolerance = 1e-6
max_iterations = 1000

# Generate non-uniform grid clustered near the walls
xi = np.linspace(0, 1, n)
stretch_factor = 3.0
y = H * (0.5 * (1 - np.tanh(stretch_factor * (1 - 2 * xi)) / np.tanh(stretch_factor)))

dy = np.diff(y)
dy_i = dy[:-1]

# Initial guess for friction velocity
u_tau = 1.0
nu = mu / rho

for iteration in range(max_iterations):
    y_plus = y * u_tau / nu
    mu_t = rho * u_tau**2 * (B * y_plus) / (1 + C * y_plus + D * y_plus**3)
    mu_eff = mu + mu_t

    mu_eff_half = (mu_eff[:-1] + mu_eff[1:]) / 2.0
    dy_half = dy

    A = np.zeros((n, n))
    b = np.full(n, -1.0)

    # Boundary conditions
    A[0, 0] = 1.0
    b[0] = 0.0
    A[-1, -1] = 1.0
    b[-1] = 0.0

    for i in range(1, n-1):
        A[i, i-1] = mu_eff_half[i-1] / dy_half[i-1] / dy[i-1]
        A[i, i] = -(mu_eff_half[i-1] / dy_half[i-1] + mu_eff_half[i] / dy_half[i]) / dy[i-1]
        A[i, i+1] = mu_eff_half[i] / dy_half[i] / dy[i-1]
        b[i] = -1.0

    U = solve(A, b)

    # Compute derivative at the first interior point
    dUdy = (U[1] - U[0]) / dy[0]
    tau_w = mu_eff[0] * dUdy
    u_tau_new = np.sqrt(tau_w / rho)

    if np.abs(u_tau_new - u_tau) < tolerance:
        break

    u_tau = u_tau_new

U_final = U
mu_t_final = mu_t

# Save the final solution
np.save('U.npy', U_final)
np.save('mu_t.npy', mu_t_final)