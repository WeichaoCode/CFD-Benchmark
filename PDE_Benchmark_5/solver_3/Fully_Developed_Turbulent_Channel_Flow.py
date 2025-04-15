import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from numpy.linalg import solve

# 1. Define parameters 
H = 1.0
ny = 100
mu = 1e-3
mu_t = 0.1*np.ones(ny)  # as per given μ_t must be an array, not a one-number variable.
dy = H/(ny-1)
y = np.linspace(0, H, ny)

# 2. Discretize the domain using finite difference approximations
# Centered difference approximation: d/dy ( [μ + μ_t] du/dy ) approx= (u_upstream - 2*u + u_downstream) / dy^2

diagonal = np.pad((mu+mu_t)/dy**2*2, (1, 1), 'constant')  # Coefficients for the main diagonal
upper = -np.pad((mu+mu_t)[:-1]/dy**2, (1, 0), 'constant')  # Coefficients for the upper diagonal
lower = -np.pad((mu+mu_t)[1:]/dy**2, (0, 1), 'constant')  # Coefficients for the lower diagonal

# 3. Assemble the coefficient matrix A and right-hand side vector b
diagonal[0] = diagonal[-1] = 1  # Dirichlet boundary conditions (u=0 at y=0 and y=H)
upper[0] = lower[-1] = 0  # No flux crosses the boundaries
A = diags([lower, diagonal, upper], [-1, 0, 1], shape=(ny, ny)).tocsc()
b = -np.ones(ny)
b[0] = b[-1] = 0  # Dirichlet boundary conditions (u=0 at y=0 and y=H)

# 4. Solve for velocity u using a linear system solver
u = solve(A.toarray(), b)

# 5. Visualize the velocity profile u(y)
plt.figure()
plt.plot(u, y, 'b', label='Velocity profile')
plt.xlabel('u')
plt.ylabel('y')
plt.legend()
plt.show()