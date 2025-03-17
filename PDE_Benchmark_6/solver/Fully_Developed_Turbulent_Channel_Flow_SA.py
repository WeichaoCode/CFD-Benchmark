import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


# Define physical properties
mu = 1.0          # molecular viscosity

# Define computational parameters
N = 1000          # number of points in y-direction
L = 1.0           # channel height
dy = L / (N - 1)  # grid spacing in y-direction

# Define mesh
y = np.linspace(0, L, N)

# Define turbulence model parameters
cv1 = 7.1
cb1 = 0.1355
cb2 = 0.622
cb3 = 2.0/3
kappa = 0.41
cw1 = cb1 / (kappa**2) + (1.0 + cb2) / cb3

# Define the turbulence model
def fv1(nu_tilde, nu):
    chi = nu_tilde / nu
    return chi**3 / (chi**3 + cv1**3)


def mu_t(rho, nu_tilde, nu):
    return rho * nu_tilde * fv1(nu_tilde, nu)


# Define linear system
d = np.ones(N)
d[-1] = 0.0
A = dia_matrix(([d, -d[:-1], -d[1:]], [0, -1, 1]), shape=(N, N))
b = np.ones(N)
b[0] = b[-1] = 0.0

# Solve for Nu_tilde
nu_tilde = spsolve(A, b)

# Compute eddy viscosity
mu_turb = mu_t(1.0, nu_tilde, mu)

# Compute effective viscosity
mu_eff = mu + mu_turb

# Define linear system for velocity
d = mu_eff[1:-1] / dy**2
A = dia_matrix(([-d[:-1], d[:-1] + d[1:], -d[1:]], [-1, 0, 1]), shape=(N-2, N-2))
b = -np.ones(N-2)

# Solve for velocity
u = np.zeros(N)
u[1:-1] = spsolve(A, b)

# Plot
plt.plot(u, y)
plt.xlabel('u')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Save results
np.save('velocity.npy', u)