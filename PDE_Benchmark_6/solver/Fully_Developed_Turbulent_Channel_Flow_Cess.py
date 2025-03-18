import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Constants
kappa = 0.42
A = 25.4
H = 1.0  # Channel height
mu = 1.0e-3  # Molecular viscosity
rho = 1.0  # Density
u_tau = 0.05  # Friction velocity
Re_tau = rho * u_tau * H / mu  # Friction Reynolds number

# Grid
ny = 100  # Number of grid points
y = np.linspace(0, H, ny)  # y-coordinates
dy = y[1] - y[0]  # Grid spacing

# Eddy viscosity model
y_plus = y * Re_tau
mu_t = mu * (0.5 * (1 + (1/9) * kappa**2 * Re_tau**2 * (2*y - y**2)**2 * (3 - 4*y + 2*y**2)**2 * 
                    (1 - np.exp(-y_plus/A))**2)**0.5 - 0.5)
mu_eff = mu + mu_t

# Discretization
a = mu_eff[:-1] / dy**2
b = -2 * mu_eff / dy**2
c = mu_eff[1:] / dy**2
A = diags([a, b, c], [-1, 0, 1], shape=(ny, ny)).tocsc()
A[0, 0] = A[-1, -1] = -1  # Boundary conditions
rhs = -np.ones(ny)
rhs[0] = rhs[-1] = 0  # Boundary conditions

# Solve
u = spsolve(A, rhs)

# Save
np.save('velocity.npy', u)

# Plot
plt.figure()
plt.plot(u, y)
plt.xlabel('Velocity')
plt.ylabel('y')
plt.title('Velocity profile')
plt.grid(True)
plt.show()