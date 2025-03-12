import numpy as np
import matplotlib.pyplot as plt

# setting the spatial and dimensional constants
Nx = 51    # number of grid points in x direction
Ny = 41    # number of grid points in y direction
x = np.linspace(0, 5, Nx)   # x direction
y = np.linspace(0, 4, Ny)   # y direction

# Creating 2D grid points
X, Y = np.meshgrid(x,y)

# Initialize the solution (temperature) array
T = np.zeros((Nx,Ny))

# Set thermal boundary conditions
T[:,0] = 10  # x=0
T[0,:] = 0   # y=4
T[:,Ny-1] = 40  # x=5
T[Nx-1,:] = 20  # y=0

# Iterate to converge to the solution
tolerance = 1e-5  
max_iterations = 500  
converged = False
iterations = 0

while not converged and iterations < max_iterations:
    T_new = T.copy()
    iterations += 1
    # loop for x and y (ignoring boundaries)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            T_new[i][j] = 0.5 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])
    if np.max(np.abs(T_new - T)) < tolerance:
        converged = True
    T = T_new.copy()

# Create a filled contour plot
plt.figure(figsize=(8,7))
cs = plt.contourf(X, Y, T.T, cmap='hot',levels=100)
plt.colorbar(cs)
plt.title('Steady State 2D Heat Equation Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()