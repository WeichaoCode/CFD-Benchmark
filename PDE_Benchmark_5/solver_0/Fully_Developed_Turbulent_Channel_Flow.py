import numpy as np
import matplotlib.pyplot as plt

# parameters
H = 1.0               # channel height
ny = 100              # number of grid points in y
mu = 1.0e-3           # kinematic viscosity 
mu_t = np.linspace(1.0e-3, 1.0e-2, ny) # eddy viscosity array

# grid setup
dy = H/(ny-1)         
y = np.linspace(0., H, ny)

# matrix A (Coefficient matrix) and rhs b
A = np.zeros((ny,ny))
b = -np.ones(ny)

for i in range(1, ny-1): 
    A[i,i-1] = mu + mu_t[i-1]
    A[i,i+1] = mu + mu_t[i+1]
    A[i,i] = -2.0 * (mu + mu_t[i])

# boundary conditions
A[0,0] = 1.0
A[-1,-1] = 1.0

# solving the linear system
u = np.linalg.solve(A, b)

# plotting velocity profile
plt.plot(u, y)
plt.xlabel('Velocity')
plt.ylabel('Height')
plt.grid(True)
plt.show()