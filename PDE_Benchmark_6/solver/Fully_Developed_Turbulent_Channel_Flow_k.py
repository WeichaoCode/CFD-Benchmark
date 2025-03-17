import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Constants
N = 100  # number of grid points
h = 1.0 / N  # grid space
mu = 1.0  # molecular viscosity
C_mu = 0.09  # turbulence model constant
C_epsilon1 = 1.44  # turbulence model constant
C_epsilon2 = 1.92  # turbulence model constant
k = 0.41  # turbulence kinetic energy
epsilon = 0.41  # turbulence dissipative rate
rho = 1.0  # fluid density
sigma_k = 1.0  # turbulence model constant
sigma_epsilon = 1.0  # turbulence model constant

# Initialize variables
y = np.linspace(0, 1, N+1)
u = np.zeros(N+1)  # mean velocity
mu_t = C_mu * rho * k**2 / epsilon  # eddy viscosity (turbulent)

# Discretize the equation
mu_eff = mu + mu_t  # effective viscosity
a = [mu_eff / h**2, -2 * mu_eff / h**2, mu_eff / h**2]  # coefficients for the linear system
f = -np.ones(N)  # RHS vector
f[0] = f[-1] = 0  # boundary condition

# Solve the linear system
u[1:-1] = spsolve(diags(a, offsets=[-1, 0, 1], shape=(N-1, N-1)), f)

# Save results
np.save('velocity_profile.npy', u)

# Plot results
plt.plot(y, u)
plt.xlabel('y')
plt.ylabel('u(y)')
plt.title('Mean velocity profile in a turbulent channel flow')
plt.grid(True)
plt.show()