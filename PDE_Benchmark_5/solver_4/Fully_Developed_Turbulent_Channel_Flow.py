import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

# Step 1: Define parameters
H = 1.0  # Channel height
ny = 100  # Number of grid points
mu = 0.01  # Kinematic viscosity
mu_t = 0.01  # Turbulent eddy viscosity

# Step 2: Discretize the domain
y = np.linspace(0, H, ny)  # grid points vector
dy = y[1] - y[0]  # grid spacing

# Step 3: Assemble the matrix A and vector b
A = np.zeros((ny,ny))  # Initialize the coefficient matrix
b = -np.ones(ny)  # Right-hand side vector
for i in range(1,ny-1):
    A[i, i-1] = mu + mu_t
    A[i, i] = -2 * (mu + mu_t)
    A[i, i+1] = mu + mu_t
A = A / dy**2  # Apply finite-difference formula

# Step 4: Apply Boundary Conditions
A[0, 0] = A[-1, -1] = 1
b[0] = b[-1] = 0

# Step 5: Solve the linear system
u = la.solve(A, b)  

# Step 6: Visualize the velocity profile
plt.figure()
plt.plot(u, y)
plt.xlabel('Velocity u')  
plt.ylabel('Channel height y')
plt.show()