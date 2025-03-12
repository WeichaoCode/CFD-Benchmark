import numpy as np
import matplotlib.pyplot as plt

# 1. Define parameters
H = 1.0    # channel height
ny = 100   # number of grid points
mu = 0.001 # kinematic viscosity
mu_t = np.ones((ny-2,)) * 0.01  # prescribed eddy viscosity profile

# 2. Discretize the domain
dy = H / (ny - 1)  # grid spacing
y = np.linspace(0, H, ny)  # vertical grid points

# 3. Assemble the coefficient matrix A and right-hand side vector b
A = np.zeros((ny-2, ny-2))
np.fill_diagonal(A, -2 * (mu + mu_t) / dy**2)
A += np.diag((mu + mu_t[:-1]) / dy**2, k=-1) + np.diag((mu + mu_t[:-1]) / dy**2, k=1)
b = -np.ones_like(y[1:-1])

# 4. Solve for velocity u using a linear system solver
u_inner = np.linalg.solve(A, b)

# apply boundary conditions: u(y=0) = u(y=H) = 0
u = np.r_[0, u_inner, 0]   

# 5. Visualize the velocity profile u(y)
plt.plot(u, y)
plt.xlabel('Velocity')
plt.ylabel('Height (y)')
plt.title('Velocity Profile')
plt.grid(True)
plt.show()