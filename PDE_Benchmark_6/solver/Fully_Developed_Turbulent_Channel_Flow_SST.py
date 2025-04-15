import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Constants
mu = 1.0e-5  # molecular viscosity
rho = 1.0  # density
dy = 0.01  # grid resolution
y = np.arange(-1, 1, dy)  # grid points
n = len(y)  # number of grid points

# Eddy viscosity model parameters
sigma_k = 1.0
sigma_omega = 1.0
beta_star = 0.09
alpha = 5/9
beta = 3/40
a1 = 0.31
F1 = 1.0
F2 = 1.0
S = 1.0  # strain rate

# Initialize variables
k = np.ones(n)  # turbulent kinetic energy
omega = np.ones(n)  # specific turbulent dissipation
mu_t = rho * k / omega  # eddy viscosity
mu_eff = mu + mu_t  # effective viscosity

# Discretize the equation
A = diags([mu_eff[:-1] / dy**2, -2 * mu_eff / dy**2, mu_eff[1:] / dy**2], [-1, 0, 1], shape=(n, n)).tocsc()
b = -np.ones(n)

# Solve the linear system
u = spsolve(A, b)

# Save the velocity profile
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/u2_Fully_Developed_Turbulent_Channel_Flow_SST.npy', u)

# Plot the velocity profile
plt.figure()
plt.plot(u, y)
plt.xlabel('Velocity')
plt.ylabel('y')
plt.title('Velocity profile in a turbulent channel flow')
plt.grid(True)
plt.show()