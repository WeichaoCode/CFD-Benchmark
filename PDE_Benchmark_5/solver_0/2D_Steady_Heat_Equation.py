import numpy as np
import matplotlib.pyplot as plt

# Domain size and physical variables
Lenx = 5.0
Leny = 4.0
Ttop = 0.0
Tbottom = 20.0
Tleft = 10.0
Tright = 40.0

# Numerical variables
nx = 50
ny = 50
deltax = Lenx/(nx-1)
deltay = Leny/(ny-1)
Tolerance = 1e-4  
T = np.zeros((nx,ny))

# Set up Initial conditions
T.fill(20) # initial guess for interior points
T[:,-1] = Ttop
T[:,0] = Tbottom
T[0,:] = Tleft
T[-1,:] = Tright

# Gauss-Seidel method
error = 1.0  
while error > Tolerance:
    T_old = T.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            T[i,j] = 0.5*(deltay**2*(T_old[i+1,j] + T_old[i-1,j]) + deltax**2*(T_old[i,j+1] + T_old[i,j-1])) / (2*(deltax**2 + deltay**2))
    error = (np.abs(T_old[:,:] - T[:,:])).max()

# Plot result
x = np.linspace(0, Lenx, nx)
y = np.linspace(Leny, 0, ny)  # inverted y axis
X, Y = np.meshgrid(x, y)
c = plt.contourf(X, Y, T, cmap='hot', levels=np.linspace(np.min(T), np.max(T), num=100))
plt.colorbar(c)
plt.title('2D Heat Equation solution')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().invert_yaxis()
plt.show()