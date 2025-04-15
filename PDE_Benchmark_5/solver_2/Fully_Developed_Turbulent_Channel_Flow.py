import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
H = 1.0 # channel height
ny = 100 # number of grid points in y-direction
mu = 1e-3 # kinematic viscosity
mu_t = 0.01 # eddy viscosity

y = np.linspace(0, H, ny) # grid points in y-direction

# Step 2: Discretize the domain using finite differences
dy = y[1] - y[0] # grid spacing in y-direction
A = np.zeros((ny,ny)) # coefficient matrix
B = -np.ones(ny) # right-hand side vector

# Step 3: Assemble the coefficient matrix A and right-hand side vector B
# Apply central difference for interior points and Dirichlet conditions at the boundaries
for i in range(1, ny-1):
    A[i, i-1] = -(mu + mu_t) / dy**2
    A[i, i+1] = -(mu + mu_t) / dy**2
    A[i, i] = 2*(mu + mu_t) / dy**2
A[0, 0] = A[-1, -1] = 1.0

# Step 4: Solve the linear system for u
u = np.linalg.solve(A, B)

# Step 5: Visualize the velocity profile
plt.figure()
plt.plot(u, y)
plt.xlabel('Velocity u')
plt.ylabel('Height y')
plt.title('Fully-developed turbulent channel flow')
plt.grid(True)
plt.show()