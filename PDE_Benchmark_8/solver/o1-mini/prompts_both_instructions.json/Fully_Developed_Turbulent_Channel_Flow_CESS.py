import numpy as np
import matplotlib.pyplot as plt

# Parameters
H = 2.0
n = 100
mu = 1e-3
kappa = 0.41
Re_tau = 180
A_param = 26

# Mesh generation with clustering near walls
xi = np.linspace(0, 1, n)
y = (H / 2) * (1 - np.cos(np.pi * xi))

# Compute y+
U_tau = Re_tau * mu / H
y_plus = y * U_tau / mu

# Compute mu_eff using the Cess algebraic turbulence model
term = (1/9) * kappa**2 * Re_tau**2 * (2*y - y**2)**2 * (3 - 4*y + 2*y**2)**2 * (1 - np.exp(-y_plus / A_param))
sqrt_term = np.sqrt(term)
mu_eff_over_mu = 0.5 * (1 + sqrt_term) - 0.5
mu_eff = mu * mu_eff_over_mu

# Initialize A matrix and b vector
A_matrix = np.zeros((n, n))
b = np.full(n, -1.0)

# Apply Dirichlet boundary conditions
A_matrix[0, 0] = 1.0
A_matrix[-1, -1] = 1.0
b[0] = 0.0
b[-1] = 0.0

# Compute mu_eff at cell faces
mu_eff_face = (mu_eff[:-1] + mu_eff[1:]) / 2

# Compute grid spacing
dy = np.diff(y)

# Fill the A matrix for internal nodes
for i in range(1, n-1):
    dy_plus = dy[i]
    dy_minus = dy[i-1]
    
    A_matrix[i, i-1] = mu_eff_face[i-1] / dy_minus
    A_matrix[i, i] = -(mu_eff_face[i] / dy_plus + mu_eff_face[i-1] / dy_minus)
    A_matrix[i, i+1] = mu_eff_face[i] / dy_plus

# Solve the linear system
u = np.linalg.solve(A_matrix, b)

# Compute laminar velocity profile
u_laminar = (H * y - y**2) / (2 * mu)

# Save the results
np.save('u.npy', u)
np.save('u_laminar.npy', u_laminar)

# Plotting (optional)
plt.plot(u, y, label='Turbulent')
plt.plot(u_laminar, y, label='Laminar', linestyle='--')
plt.xlabel('Velocity')
plt.ylabel('y')
plt.legend()
plt.savefig('velocity_profile.png')