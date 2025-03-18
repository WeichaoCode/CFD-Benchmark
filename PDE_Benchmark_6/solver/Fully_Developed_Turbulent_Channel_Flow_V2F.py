import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Constants
mu = 1.0e-5  # molecular viscosity
rho = 1.0  # density
L = 1.0  # channel height
N = 100  # number of grid points
dy = L / (N - 1)  # grid resolution

# Grid
y = np.linspace(0, L, N)

# Eddy viscosity model parameters
C_mu = 0.09
sigma_k = 1.0
sigma_eps = 1.3
C_eps1 = 1.44
C_eps2 = 1.92

# Turbulence quantities
k = np.ones(N)  # turbulent kinetic energy
eps = np.ones(N)  # turbulent dissipation rate
v2 = np.ones(N)  # wall-normal fluctuation
f = np.ones(N)  # elliptic relaxation function
T_t = 1.0  # turbulent time scale

# Eddy viscosity
mu_t = C_mu * rho * v2 * T_t

# Effective viscosity
mu_eff = mu + mu_t

# Finite difference coefficients
a = mu_eff[:-2] / dy**2
b = -2 * mu_eff[1:-1] / dy**2
c = mu_eff[2:] / dy**2

# Assemble the coefficient matrix
A = diags([a, b, c], [-1, 0, 1], shape=(N-2, N-2)).tocsc()

# Right-hand side vector
rhs = -np.ones(N-2)

# Solve the linear system
u = np.zeros(N)
u[1:-1] = spsolve(A, rhs)

# Save the velocity profile
np.save('velocity_profile.npy', u)

# Plot the velocity profile
plt.figure()
plt.plot(u, y)
plt.xlabel('Velocity')
plt.ylabel('y')
plt.title('Velocity profile in a turbulent channel flow')
plt.grid(True)
plt.show()