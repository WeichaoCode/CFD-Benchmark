import numpy as np
import matplotlib.pyplot as plt

#  Step 1: Define parameters
H = 2.0 # channel height
ny = 100 # number of grid cells
mu = 0.01 # kinematic viscosity
mu_t = 0.05 # eddy viscosity
dy = H / ny # grid cell size

#  Step 2: Discretize the domain
y = np.linspace(0 + dy / 2, H - dy / 2, ny) # node locations

#  Step 3: Assemble the coefficient matrix A and right-hand side vector b
A = np.zeros((ny, ny))
b = -np.ones(ny)

for i in range(ny):
    if i == 0 or i == ny-1:
        A[i, i] = 1.0 # boundary condition
    else:
        A[i, i-1] = -(mu + mu_t) / dy**2
        A[i, i]   = 2*(mu + mu_t) / dy**2
        A[i, i+1] = -(mu + mu_t) / dy**2

#  Step 4: Solve for velocity u
u = np.linalg.solve(A, b)

#  Step 5: Visualize the velocity profile
plt.figure()
plt.plot(u, y)
plt.xlabel('velocity') 
plt.ylabel('height')
plt.title('Velocity profile')
plt.grid()
plt.show()